#!/usr/bin/env python3
"""Rewrite openWakeWord's embedding model to take 8 new mel frames per call.

The stock model takes a 76-frame window and is called every 80 ms on a window
that has moved 8 frames, so 68 of 76 frames are recomputed each time. Its
convolutions are unpadded in time and its pooling has a total stride of 8, so
the same outputs can be had by keeping the tail of a layer's input from the
previous call.

    oww_stream.py embedding_model.onnx out.onnx [--stock] [--no-clip]
                  [--per-block] [--nchw-states]

Inputs:  mel [1, 8, 32, 1], state_0..state_N (zeros to start)
Outputs: embedding [1, 96], state_0_next..state_N_next
The first 9 calls are warm-up; from the 10th the output equals the stock
model's on the last 76 frames fed.

Three choices, each measured on an Echo (device/tools/oww_stream/README.md).
The defaults are the fastest of each:

  activation   The model's activation is Max(LeakyRelu(x, 0.2), -0.4), two
               nodes the accelerated backend does not run. It equals
               LeakyRelu(Clip(x, min=-2)), and a Clip after a Conv is fused
               into the Conv. --no-clip keeps the original pair.
  states       One state per convolution that looks across time (11).
               --per-block keeps one per block of convolutions between time
               poolings (4): fewer nodes, more rows recomputed, and slower.
  layout       States cross the model boundary in NHWC, which is the layout
               the backend computes in, so no Transpose runs per state.
               --nchw-states keeps them NCHW.

--stock applies only the activation rewrite and keeps the 76-frame input: a
drop-in replacement for the stock file.
"""
import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

STEP = 8


def _const(name, values, dtype=np.int64):
    return numpy_helper.from_array(np.array(values, dtype), name)


def _attr(node, name):
    return next(a for a in node.attribute if a.name == name)


def rewrite_activation(g, inits):
    """Max(LeakyRelu(x, a), c)  ->  LeakyRelu(Clip(x, min=c/a), a)."""
    by_out = {n.output[0]: n for n in g}
    consts = {i.name: numpy_helper.to_array(i) for i in inits}
    out, done = [], 0
    drop = set()
    for n in g:
        if n.op_type == "Max" and n.input[0] in by_out and by_out[n.input[0]].op_type == "LeakyRelu":
            leaky = by_out[n.input[0]]
            alpha = np.float32(_attr(leaky, "alpha").f)
            c = np.float32(consts[n.input[1]].ravel()[0])
            lo = np.float32(c / alpha)
            if np.float32(lo * alpha) != c or consts[n.input[1]].size != 1:
                raise SystemExit(f"activation does not rewrite exactly: {c} / {alpha}")
            lo_name, clip_out = f"clip_min_{done}", f"clip_{done}"
            inits.append(_const(lo_name, lo, np.float32))
            out.remove(leaky)
            out.append(helper.make_node("Clip", [leaky.input[0], lo_name], [clip_out]))
            out.append(helper.make_node("LeakyRelu", [clip_out], [n.output[0]], alpha=float(alpha)))
            drop.add(n.input[1])
            done += 1
            continue
        out.append(n)
    inits[:] = [i for i in inits if i.name not in drop]
    return out, done


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--stock", action="store_true")
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--per-block", action="store_true")
    ap.add_argument("--nchw-states", action="store_true")
    ap.add_argument("--perturb", action="store_true",
                    help="scale one weight tensor: a streaming model that is WRONG, for testing the firmware's self-check")
    a = ap.parse_args()

    m = onnx.load(a.src)
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in shape_inference.infer_shapes(m).graph.value_info}
    g = m.graph
    first = next(n for n in g.node if n.op_type == "Reshape" and n.input[0] == g.input[0].name)
    last = next(n for n in g.node if n.op_type == "Reshape" and n.output[0] == g.output[0].name)
    inits = list(g.initializer)
    body = [n for n in g.node if n is not first and n is not last]
    clipped = 0
    if not a.no_clip:
        body, clipped = rewrite_activation(body, inits)

    if a.stock:
        graph = helper.make_graph([first] + body + [last], "oww_embedding", list(g.input), list(g.output), inits)
        out = helper.make_model(graph, opset_imports=m.opset_import, ir_version=m.ir_version)
        onnx.checker.check_model(out)
        onnx.save(out, a.dst)
        print(f"{a.dst}: stock shape, {clipped} activations rewritten")
        return

    if a.perturb:
        big = max((i for i in inits if len(i.dims) == 4), key=lambda i: int(np.prod(i.dims)))
        inits[inits.index(big)] = numpy_helper.from_array(numpy_helper.to_array(big) * 1.05, big.name)
    inits = [i for i in inits if i.name not in (first.input[1], last.input[1])]
    inits += [_const("stream_in_shape", [1, 1, STEP, 32]), _const("stream_out_shape", [1, 96]),
              _const("tail_end", [2 ** 31]), _const("time_axis", [2])]

    def time_kernel(n):
        return n.op_type == "Conv" and _attr(n, "kernel_shape").ints[0] == 3

    def time_pool(n):
        return n.op_type == "MaxPool" and _attr(n, "strides").ints[0] == 2

    # Where a state goes: before each time-convolution, or before the first
    # one of each block between time poolings, carrying 2 rows per
    # time-convolution it has to feed.
    points, rows = [], []
    if not a.per_block:
        for i, n in enumerate(body):
            if time_kernel(n):
                points.append(i)
                rows.append(2)
    else:
        start, count = None, 0
        for i, n in enumerate(body):
            if time_kernel(n):
                if start is None:
                    start = i
                count += 1
            if time_pool(n) and start is not None:
                points.append(start)
                rows.append(2 * count)
                start, count = None, 0
        if start is not None:
            points.append(start)
            rows.append(2 * count)

    nodes = [helper.make_node("Reshape", ["mel", "stream_in_shape"], [first.output[0]])]
    ins = [helper.make_tensor_value_info("mel", TensorProto.FLOAT, [1, STEP, 32, 1])]
    outs = [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, 96])]
    floats = 0
    for i, n in enumerate(body):
        if i in points:
            k = points.index(i)
            r = rows[k]
            x = n.input[0]
            c, w = (1, 32) if x == first.output[0] else (shapes[x][1], shapes[x][3])
            st_in, st_out, cat = f"state_{k}", f"state_{k}_next", f"stream_cat_{k}"
            inits.append(_const(f"tail_start_{k}", [-r]))
            floats += c * r * w
            if a.nchw_states:
                shape = [1, c, r, w]
                nodes.append(helper.make_node("Concat", [st_in, x], [cat], axis=2))
                nodes.append(helper.make_node("Slice", [cat, f"tail_start_{k}", "tail_end", "time_axis"], [st_out]))
            else:
                shape = [1, r, w, c]
                nodes.append(helper.make_node("Transpose", [st_in], [st_in + "_nchw"], perm=[0, 3, 1, 2]))
                nodes.append(helper.make_node("Concat", [st_in + "_nchw", x], [cat], axis=2))
                nodes.append(helper.make_node("Slice", [cat, f"tail_start_{k}", "tail_end", "time_axis"], [st_out + "_nchw"]))
                nodes.append(helper.make_node("Transpose", [st_out + "_nchw"], [st_out], perm=[0, 2, 3, 1]))
            ins.append(helper.make_tensor_value_info(st_in, TensorProto.FLOAT, shape))
            outs.append(helper.make_tensor_value_info(st_out, TensorProto.FLOAT, shape))
            n.input[0] = cat
        nodes.append(n)
    nodes.append(helper.make_node("Reshape", [last.input[0], "stream_out_shape"], ["embedding"]))
    graph = helper.make_graph(nodes, "oww_embedding_stream", ins, outs, inits)
    out = helper.make_model(graph, opset_imports=m.opset_import, ir_version=m.ir_version)
    onnx.checker.check_model(out)
    onnx.save(out, a.dst)
    print(f"{a.dst}: {len(points)} states ({floats} floats), {clipped} activations rewritten, "
          f"{'NCHW' if a.nchw_states else 'NHWC'} states")


if __name__ == "__main__":
    main()
