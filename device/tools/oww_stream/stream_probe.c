/* stream_probe: stock vs streaming openWakeWord embedding on the device.
 *
 *   stream_probe LIB STOCK.onnx seconds xnnpack(0|1) MODEL.onnx...
 *
 * Each MODEL (stock shape, or streaming with any number of states) is fed the
 * same mel frames as STOCK and its embeddings compared, then run at the real
 * duty cycle (one call per 80 ms) and its process CPU reported, in the lines
 * device/tools/oww_probe prints.
 * Session options are the firmware's: 1 thread, ORT_ENABLE_ALL, no spinning.
 */
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include "onnxruntime_c_api.h"

#define NSTATE 11
#define STEP 8
#define WIN 76
#define MELS 32
#define EMB 96

static const OrtApi *api;
static void die(const char *what, OrtStatus *st) {
	fprintf(stderr, "FAIL %s: %s\n", what, st ? api->GetErrorMessage(st) : "");
	exit(1);
}
#define CK(what, call) do { OrtStatus *s_ = (call); if (s_) die(what, s_); } while (0)

static double now_ms(void) {
	struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
static double cpu_s(void) {
	struct rusage r; getrusage(RUSAGE_SELF, &r);
	return r.ru_utime.tv_sec + r.ru_utime.tv_usec / 1e6 + r.ru_stime.tv_sec + r.ru_stime.tv_usec / 1e6;
}
static uint64_t seed = 0x9E3779B97F4A7C15ULL;
static float rnd(void) {            /* uint64_t: unsigned long is 32-bit here */
	seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
	return (float)((seed >> 33) % 2000000) / 1000000.0f * 6.0f - 6.0f;
}

static OrtSession *open_model(OrtEnv *env, const char *path, int xnn, int *xnn_on) {
	OrtSessionOptions *o; CK("opts", api->CreateSessionOptions(&o));
	CK("threads", api->SetIntraOpNumThreads(o, 1));
	CK("opt", api->SetSessionGraphOptimizationLevel(o, ORT_ENABLE_ALL));
	CK("spin", api->AddSessionConfigEntry(o, "session.intra_op.allow_spinning", "0"));
	CK("spin", api->AddSessionConfigEntry(o, "session.inter_op.allow_spinning", "0"));
	*xnn_on = 0;
	if (xnn) {
		const char *k[] = {"intra_op_num_threads"}, *v[] = {"1"};
		OrtStatus *st = api->SessionOptionsAppendExecutionProvider(o, "XNNPACK", k, v, 1);
		if (st) api->ReleaseStatus(st); else *xnn_on = 1;
	}
	OrtSession *s; CK(path, api->CreateSession(env, path, o, &s));
	api->ReleaseSessionOptions(o);
	return s;
}

static OrtMemoryInfo *mi;
static OrtValue *tensor(float *buf, size_t n, const int64_t *shape, size_t nd) {
	OrtValue *v; CK("tensor", api->CreateTensorWithDataAsOrtValue(mi, buf, n * sizeof(float), shape, nd,
	                                                             ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &v));
	return v;
}

/* stock: one 76-frame window in, one embedding out */
static void run_stock(OrtSession *s, float *win, float *emb) {
	int64_t sh[4] = {1, WIN, MELS, 1};
	OrtValue *in = tensor(win, WIN * MELS, sh, 4), *out = NULL;
	const char *inn[] = {"input_1"}, *outn[] = {"conv2d_19"};
	CK("run stock", api->Run(s, NULL, inn, (const OrtValue *const *)&in, 1, outn, 1, &out));
	float *p; CK("data", api->GetTensorMutableData(out, (void **)&p));
	memcpy(emb, p, EMB * sizeof(float));
	api->ReleaseValue(in); api->ReleaseValue(out);
}

/* streaming: 8 new frames and the states in, embedding and next states out */
#define MAXS 16
typedef struct {
	OrtSession *s; int n;                       /* n states; 0 = stock shape */
	char in_names[MAXS + 1][32], out_names[MAXS + 1][32];
	const char *inp[MAXS + 1], *outp[MAXS + 1];
	float *state[MAXS]; size_t state_n[MAXS]; int64_t shape[MAXS][4];
} model;

static void model_init(model *m, OrtSession *s) {
	OrtAllocator *al; CK("alloc", api->GetAllocatorWithDefaultOptions(&al));
	size_t n; CK("count", api->SessionGetInputCount(s, &n));
	m->s = s; m->n = (int)n - 1;
	if (m->n > MAXS) { fprintf(stderr, "FAIL %zu inputs\n", n); exit(1); }
	if (m->n == 0) return;
	strcpy(m->in_names[0], "mel"); strcpy(m->out_names[0], "embedding");
	for (size_t i = 0; i < n; i++) {
		char *name; CK("name", api->SessionGetInputName(s, i, al, &name));
		int k = -1;
		if (sscanf(name, "state_%d", &k) == 1 && k >= 0 && k < m->n) {
			OrtTypeInfo *ti; CK("ti", api->SessionGetInputTypeInfo(s, i, &ti));
			const OrtTensorTypeAndShapeInfo *tsi; CK("cast", api->CastTypeInfoToTensorInfo(ti, &tsi));
			CK("dims", api->GetDimensions(tsi, m->shape[k], 4));
			m->state_n[k] = (size_t)(m->shape[k][0] * m->shape[k][1] * m->shape[k][2] * m->shape[k][3]);
			m->state[k] = calloc(m->state_n[k], sizeof(float));
			snprintf(m->in_names[k + 1], 32, "state_%d", k);
			snprintf(m->out_names[k + 1], 32, "state_%d_next", k);
			api->ReleaseTypeInfo(ti);
		}
		al->Free(al, name);
	}
	for (int i = 0; i <= m->n; i++) { m->inp[i] = m->in_names[i]; m->outp[i] = m->out_names[i]; }
}
static void model_reset(model *m) { for (int k = 0; k < m->n; k++) memset(m->state[k], 0, m->state_n[k] * sizeof(float)); }

/* one step: `mel` points at the newest 8 frames, with 68 older ones before it */
static void step(model *m, float *mel, float *emb) {
	if (m->n == 0) { run_stock(m->s, mel - (WIN - STEP) * MELS, emb); return; }
	int64_t sh[4] = {1, STEP, MELS, 1};
	OrtValue *in[MAXS + 1], *out[MAXS + 1];
	in[0] = tensor(mel, STEP * MELS, sh, 4);
	for (int k = 0; k < m->n; k++) in[k + 1] = tensor(m->state[k], m->state_n[k], m->shape[k], 4);
	memset(out, 0, sizeof(out));
	CK("run stream", api->Run(m->s, NULL, m->inp, (const OrtValue *const *)in, m->n + 1, m->outp, m->n + 1, out));
	float *p; CK("data", api->GetTensorMutableData(out[0], (void **)&p));
	memcpy(emb, p, EMB * sizeof(float));
	for (int k = 0; k < m->n; k++) {
		CK("state", api->GetTensorMutableData(out[k + 1], (void **)&p));
		memcpy(m->state[k], p, m->state_n[k] * sizeof(float));
	}
	for (int i = 0; i <= m->n; i++) { api->ReleaseValue(in[i]); api->ReleaseValue(out[i]); }
}

static int cmp(const void *a, const void *b) { double d = *(const double *)a - *(const double *)b; return (d > 0) - (d < 0); }

static void paced(const char *label, model *m, float *mel, int steps, int seconds) {
	int frames = seconds * 1000 / 80, late = 0;
	double *lat = malloc(frames * sizeof(double));
	float emb[EMB];
	double c0 = cpu_s(), t0 = now_ms();
	for (int i = 0; i < frames; i++) {
		double due = t0 + i * 80.0, a = now_ms();
		step(m, mel + (10 + i % (steps - 10)) * STEP * MELS, emb);
		lat[i] = now_ms() - a;
		if (lat[i] > 80.0) late++;
		double left = due + 80.0 - now_ms();
		if (left > 0) { struct timespec ts = {(time_t)(left / 1000), (long)(fmod(left, 1000) * 1e6)}; nanosleep(&ts, NULL); }
	}
	double wall = (now_ms() - t0) / 1000, cpu = cpu_s() - c0;
	qsort(lat, frames, sizeof(double), cmp);
	printf("  frames     %d over %.1fs (%d overran their 80ms slot)\n", frames, wall, late);
	printf("  latency    p50 %.1fms  p95 %.1fms  max %.1fms  (budget 80ms)\n",
	       lat[frames / 2], lat[frames * 95 / 100], lat[frames - 1]);
	printf("  CPU        %.1f%% of one core  (%.2fs of CPU in %.1fs wall)\n", 100 * cpu / wall, cpu, wall);
	free(lat);
	(void)label;
}

int main(int argc, char **argv) {
	if (argc < 6) { fprintf(stderr, "usage: stream_probe LIB STOCK.onnx seconds xnnpack(0|1) MODEL.onnx...\n"); return 2; }
	int seconds = atoi(argv[3]), xnn = atoi(argv[4]);
	void *dl = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
	if (!dl) { fprintf(stderr, "FAIL dlopen: %s\n", dlerror()); return 1; }
	const OrtApiBase *(*base)(void) = (const OrtApiBase *(*)(void))dlsym(dl, "OrtGetApiBase");
	api = base()->GetApi(ORT_API_VERSION);
	if (!api) { fprintf(stderr, "FAIL runtime too old: %s\n", base()->GetVersionString()); return 1; }
	OrtEnv *env; CK("env", api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "probe", &env));
	CK("mi", api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));
	int x;
	OrtSession *stock = open_model(env, argv[2], xnn, &x);
	printf("runtime   onnxruntime %s\nsession   threads=1 xnnpack=%d(requested %d) spinning=0\n",
	       base()->GetVersionString(), x, xnn);

	enum { STEPS = 60 };
	static float mel[STEPS * STEP * MELS];
	for (size_t i = 0; i < sizeof(mel) / sizeof(mel[0]); i++) mel[i] = rnd();
	static float ref[STEPS][EMB];
	for (int j = 9; j < STEPS; j++) run_stock(stock, mel + ((j + 1) * STEP - WIN) * MELS, ref[j]);

	for (int a = 5; a < argc; a++) {
		model m; memset(&m, 0, sizeof(m));
		model_init(&m, open_model(env, argv[a], xnn, &x));
		double worst = 0; float e[EMB];
		model_reset(&m);
		for (int j = 0; j < STEPS; j++) {
			if (m.n == 0 && j < 9) continue;
			step(&m, mel + j * STEP * MELS, e);
			if (j < 9) continue;
			for (int k = 0; k < EMB; k++) { double d = fabs((double)e[k] - ref[j][k]); if (d > worst) worst = d; }
		}
		const char *nm = strrchr(argv[a], '/'); nm = nm ? nm + 1 : argv[a];
		printf("\n== %s: %d states, max abs diff from stock %.3g ==\n", nm, m.n, worst);
		model_reset(&m);
		paced(nm, &m, mel, STEPS, seconds);
	}
	return 0;
}
