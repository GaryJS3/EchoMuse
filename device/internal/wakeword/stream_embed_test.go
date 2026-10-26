package wakeword

import (
	"errors"
	"math"
	"testing"
)

// fakeModels stands in for the three models with functions whose output
// depends on exactly the input they were given, so a Detector that feeds a
// wrong frame, in a wrong order, or from before a reset gets a different
// embedding rather than a plausible one.
type fakeModels struct {
	next   float32 // every mel value ever produced is distinct
	embeds int
}

func (f *fakeModels) Melspec(samples []float32) ([]float32, int, error) {
	frames := int(math.Ceil(float64(len(samples))/160.0 - 3.0))
	if frames < 0 {
		frames = 0
	}
	out := make([]float32, frames*MelBins)
	for i := range out {
		f.next++
		out[i] = f.next
	}
	return out, frames, nil
}

// windowEmbedding is a function of every frame in the window and its place.
func windowEmbedding(window []float32) []float32 {
	out := make([]float32, FeatDim)
	for i, v := range window {
		out[i%FeatDim] += v * float32(1+i/MelBins)
	}
	return out
}

func (f *fakeModels) Embed(window []float32) ([]float32, error) {
	f.embeds++
	return windowEmbedding(window), nil
}
func (f *fakeModels) Classify(feats []float32) (float32, error) { return 0, nil }

// fakeStream behaves as the streaming model does: its output is the stock
// embedding of the last MelWindow frames it was fed, and is only valid once
// streamPrime steps have been fed since the last reset.
type fakeStream struct {
	fakeModels
	off     bool
	fed     []float32
	steps   int // since reset
	calls   int
	resets  int
	failAt  int // 1-based call number that returns an error; 0 = never
	invalid int // outputs handed back before the state was primed
}

func (f *fakeStream) Streaming() bool { return !f.off }
func (f *fakeStream) ResetStream()    { f.fed, f.steps = nil, 0; f.resets++ }
func (f *fakeStream) EmbedStep(frames []float32) ([]float32, error) {
	f.calls++
	if f.calls == f.failAt {
		return nil, errors.New("fake stream failure")
	}
	if len(frames) != MelStep*MelBins {
		return nil, errors.New("wrong step size")
	}
	f.fed = append(f.fed, frames...)
	f.steps++
	if f.steps < streamPrime {
		f.invalid++
		return make([]float32, FeatDim), nil
	}
	return windowEmbedding(f.fed[len(f.fed)-MelWindow*MelBins:]), nil
}

func pushAll(t *testing.T, d *Detector, sizes ...int) {
	t.Helper()
	for _, n := range sizes {
		if _, err := d.Push(make([]int16, n)); err != nil {
			t.Fatal(err)
		}
	}
}

func sameFeatures(t *testing.T, stock, stream *Detector) {
	t.Helper()
	if len(stock.feat) != len(stream.feat) {
		t.Fatalf("stock holds %d feature values, streaming %d", len(stock.feat), len(stream.feat))
	}
	for i := range stock.feat {
		if stock.feat[i] != stream.feat[i] {
			t.Fatalf("embedding %d value %d: stock %v, streaming %v",
				i/FeatDim, i%FeatDim, stock.feat[i], stream.feat[i])
		}
	}
}

// The sizes cover what the device does and what it could: single chunks, the
// 160ms batches the mic delivers, a ragged tail, and a first chunk that
// yields 5 mel frames where every later one yields 8.
var pushSizes = []int{
	ChunkSamples, ChunkSamples, 2 * ChunkSamples, 2*ChunkSamples + 37, ChunkSamples - 37,
	3 * ChunkSamples, ChunkSamples, 5, ChunkSamples - 5, 2 * ChunkSamples,
}

func TestStreamingEmbeddingsEqualStock(t *testing.T) {
	stock, fs := New(&fakeModels{}), &fakeStream{}
	stream := New(fs)
	pushAll(t, stock, pushSizes...)
	pushAll(t, stream, pushSizes...)
	sameFeatures(t, stock, stream)
	if fs.embeds != 0 {
		t.Fatalf("streaming detector ran the stock embedding %d times", fs.embeds)
	}
}

// Once primed, a chunk costs one step. A Detector that re-primed every chunk
// would still be correct and would cost more than the stock model.
func TestSteadyStateIsOneStepPerChunk(t *testing.T) {
	fs := &fakeStream{}
	d := New(fs)
	pushAll(t, d, ChunkSamples, ChunkSamples)
	before, resets := fs.calls, fs.resets
	for i := 0; i < 50; i++ {
		pushAll(t, d, ChunkSamples)
	}
	if got := fs.calls - before; got != 50 {
		t.Fatalf("50 chunks took %d steps, want 50", got)
	}
	if fs.resets != resets {
		t.Fatalf("the stream was reset %d times in steady state", fs.resets-resets)
	}
}

// The first chunk adds 5 mel frames where every later one adds 8. The state
// is built for wherever the first window ends, so that needs no special case:
// one priming, then one step per chunk.
func TestTheShortFirstChunkNeedsNoSecondPriming(t *testing.T) {
	fs := &fakeStream{}
	d := New(fs)
	pushAll(t, d, ChunkSamples)
	if frames := len(d.mel)/MelBins - MelWindow; frames != 5 {
		t.Fatalf("first chunk added %d mel frames; this test expects the 5 the real model gives", frames)
	}
	if fs.calls != streamPrime || fs.resets != 1 {
		t.Fatalf("first window: %d steps, %d resets; want %d and 1", fs.calls, fs.resets, streamPrime)
	}
	pushAll(t, d, ChunkSamples)
	if fs.calls != streamPrime+1 || fs.resets != 1 {
		t.Fatalf("second window: %d steps, %d resets; want %d and 1", fs.calls, fs.resets, streamPrime+1)
	}
}

// A chunk that adds some other number of frames moves the window off the
// stream's grid. The state is rebuilt from the ring, never stepped anyway.
func TestAnIrregularChunkRebuildsTheState(t *testing.T) {
	stock, fs := New(&fakeModels{}), &fakeStream{}
	stream := New(fs)
	for _, d := range []*Detector{stock, stream} {
		pushAll(t, d, ChunkSamples, ChunkSamples, ChunkSamples)
		// As a melspec call returning 3 frames short would leave it.
		d.mel = d.mel[:len(d.mel)-3*MelBins]
		d.melTotal -= 3
		pushAll(t, d, ChunkSamples, ChunkSamples)
	}
	sameFeatures(t, stock, stream)
	if fs.resets != 2 {
		t.Fatalf("stream reset %d times, want 2: once at the start, once for the irregular chunk", fs.resets)
	}
}

func TestResetRebuildsTheStreamState(t *testing.T) {
	stock, fs := New(&fakeModels{}), &fakeStream{}
	stream := New(fs)
	for _, d := range []*Detector{stock, stream} {
		pushAll(t, d, pushSizes...)
		d.Reset()
		pushAll(t, d, pushSizes...)
	}
	sameFeatures(t, stock, stream)
}

// The mel ring is trimmed at MelBufMax frames. Positions are counted from the
// start of the stream, so the trim must not look like a jump.
func TestRingTrimDoesNotDisturbTheStream(t *testing.T) {
	stock, fs := New(&fakeModels{}), &fakeStream{}
	stream := New(fs)
	chunks := MelBufMax/MelStep + 40
	for i := 0; i < chunks; i++ {
		pushAll(t, stock, ChunkSamples)
		pushAll(t, stream, ChunkSamples)
	}
	if len(stream.mel)/MelBins != MelBufMax {
		t.Fatalf("ring holds %d frames, expected it trimmed to %d", len(stream.mel)/MelBins, MelBufMax)
	}
	if fs.resets != 1 {
		t.Fatalf("the stream was reset %d times over a trimmed ring, want only the one at the start", fs.resets)
	}
	// FeatBufMax trims both the same way, so the rings stay comparable.
	sameFeatures(t, stock, stream)
}

func TestAStreamThatIsOffIsNeverCalled(t *testing.T) {
	stock, fs := New(&fakeModels{}), &fakeStream{off: true}
	stream := New(fs)
	pushAll(t, stock, pushSizes...)
	pushAll(t, stream, pushSizes...)
	sameFeatures(t, stock, stream)
	if fs.calls != 0 {
		t.Fatalf("EmbedStep was called %d times on a stream that is off", fs.calls)
	}
}

// A failing stream costs CPU, never an embedding: the window it failed on is
// embedded by the stock model, and the next one rebuilds the state.
func TestAFailedStepFallsBackOnTheSameWindow(t *testing.T) {
	for _, failAt := range []int{1, streamPrime, streamPrime + 3} {
		stock, fs := New(&fakeModels{}), &fakeStream{failAt: failAt}
		stream := New(fs)
		pushAll(t, stock, pushSizes...)
		pushAll(t, stream, pushSizes...)
		sameFeatures(t, stock, stream)
		if fs.embeds != 1 {
			t.Fatalf("failure at step %d: stock embedding ran %d times, want 1", failAt, fs.embeds)
		}
	}
}

func TestPrimeCoversTheWindow(t *testing.T) {
	if streamPrime*MelStep < MelWindow || (streamPrime-1)*MelStep >= MelWindow {
		t.Fatalf("streamPrime %d does not just cover %d frames in steps of %d", streamPrime, MelWindow, MelStep)
	}
}
