"""Export miro-2b (merged HF, llama arch) to a single ONNX decoding graph with
flattened KV cache, then weight-quantise MatMuls to int4 (RTN, block=128, symmetric).

Why hand-rolled instead of optimum / onnxruntime-genai builder:
- optimum's exporter forks a child that loads the 5GB bf16 weights a second time
- locally the genai builder grew to ~9GB RSS and was OOM-killed inside the 8GB
  cgroup even at 1% progress; this script keeps exactly ONE copy of the weights
  alive inside a single process.

I/O contract (what the app has to feed):
  input_ids            [1, S]        int64
  attention_mask       [1, P + S]    int64   (1 = real token, 0 = pad)
  position_ids         [1, S]        int64   (absolute, = P + arange(S))
  past_key_values.i.key   [1, 2, P, 128]  float16
  past_key_values.i.value [1, 2, P, 128]  float16
  -> logits            [1, S, 130560]
  -> present.i.key / present.i.value  [1, 2, P + S, 128]
"""
import argparse
import gc
import json
import os
import resource
import shutil
import sys
import time

import torch


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def unpack_pkv(pkv):
    """transformers 5.22: cache.layers[i].keys/.values; 4.x: tuple of (k, v)."""
    layers = getattr(pkv, "layers", None)
    if layers is not None:
        return [(getattr(l, "keys"), getattr(l, "values")) for l in layers]
    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        return list(zip(pkv.key_cache, pkv.value_cache))
    return list(pkv)


def make_cache(ks, vs):
    """transformers 5.x needs a Cache object; 4.x accepts a tuple of tuples."""
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return tuple(zip(ks, vs))
    try:
        cache = DynamicCache()
        for i, (k, v) in enumerate(zip(ks, vs)):
            cache.update(k, v, i)
        return cache
    except Exception:
        try:
            return DynamicCache(ddp_cache_data=list(zip(ks, vs)))
        except Exception:
            return tuple(zip(ks, vs))


class Decoder(torch.nn.Module):
    def __init__(self, model, n_layers):
        super().__init__()
        self.model = model
        self.n_layers = n_layers

    def forward(self, input_ids, attention_mask, position_ids, *past):
        ks = [past[2 * i] for i in range(self.n_layers)]
        vs = [past[2 * i + 1] for i in range(self.n_layers)]
        pkv = make_cache(ks, vs)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=pkv,
            use_cache=True,
            return_dict=True,
        )
        flat = []
        for k, v in unpack_pkv(out.past_key_values):
            flat.append(k)
            flat.append(v)
        return (out.logits, *flat)


def cmd_export(args):
    from transformers import AutoConfig, AutoModelForCausalLM

    t0 = time.time()
    cfg = AutoConfig.from_pretrained(args.model)
    n_layers = int(cfg.num_hidden_layers)
    n_kv = int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    print(f"layers={n_layers} kv_heads={n_kv} head_dim={head_dim} vocab={cfg.vocab_size}", flush=True)

    torch.set_grad_enabled(False)
    print("loading model (fp16, cpu, single copy) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, low_cpu_mem_usage=True, device_map=None
    )
    model.eval()
    model.config.use_cache = True
    print(f"loaded  rss={rss_gb():.2f}GB  ({time.time()-t0:.0f}s)", flush=True)

    print(f"loaded fp16 rss={rss_gb():.2f}GB", flush=True)

    wrapper = Decoder(model, n_layers)

    B, S, P = 1, 4, 8
    input_ids = torch.zeros((B, S), dtype=torch.long)
    attention_mask = torch.ones((B, P + S), dtype=torch.long)
    position_ids = (torch.arange(P, P + S, dtype=torch.long)).unsqueeze(0)
    past = []
    for _ in range(n_layers):
        past.append(torch.zeros((B, n_kv, P, head_dim), dtype=torch.float16))
        past.append(torch.zeros((B, n_kv, P, head_dim), dtype=torch.float16))
    dummy = (input_ids, attention_mask, position_ids, *past)

    in_names = ["input_ids", "attention_mask", "position_ids"]
    for i in range(n_layers):
        in_names.append(f"past_key_values.{i}.key")
        in_names.append(f"past_key_values.{i}.value")
    out_names = ["logits"]
    for i in range(n_layers):
        out_names.append(f"present.{i}.key")
        out_names.append(f"present.{i}.value")

    dynamic = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "total_seq"},
        "position_ids": {0: "batch", 1: "seq"},
        "logits": {0: "batch", 1: "seq"},
    }
    for i in range(n_layers):
        dynamic[f"past_key_values.{i}.key"] = {0: "batch", 2: "past_seq"}
        dynamic[f"past_key_values.{i}.value"] = {0: "batch", 2: "past_seq"}
        dynamic[f"present.{i}.key"] = {0: "batch", 2: "total_seq"}
        dynamic[f"present.{i}.value"] = {0: "batch", 2: "total_seq"}

    # Smoke-test the wrapper BEFORE deleting the checkpoint - every failed attempt
    # would otherwise cost a 4.7G re-download.
    print("smoke forward ...", flush=True)
    with torch.no_grad():
        _ = wrapper(*dummy)
    print("smoke forward ok", flush=True)

    # 10G sandbox: weights are in RAM now, so free the checkpoint right before we
    # write the fp16 graph (never earlier - a failed export would force a re-download
    # of the 4.7G merged model).
    if os.environ.get("DELETE_SRC", "0") == "1" and os.path.isdir(args.model):
        shutil.rmtree(args.model)
        gc.collect()
        print("deleted source checkpoint (weights are in RAM now)", flush=True)
        import subprocess

        print(subprocess.run(["df", "-h", "/"], capture_output=True, text=True).stdout, flush=True)

    os.makedirs(args.work, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    fp16_path = os.path.join(args.work, "model.fp16.onnx")
    print("torch.onnx.export (this takes a few minutes) ...", flush=True)
    torch.onnx.export(
        wrapper,
        dummy,
        fp16_path,
        input_names=in_names,
        output_names=out_names,
        dynamic_axes=dynamic,
        opset_version=17,
        do_constant_folding=False,
        dynamo=False,
        external_data=True,
        optimize=False,
    )
    print(f"exported rss={rss_gb():.2f}GB  ({time.time()-t0:.0f}s)", flush=True)
    size = os.path.getsize(fp16_path) + sum(
        os.path.getsize(os.path.join(args.work, f))
        for f in os.listdir(args.work)
        if f.startswith("model.fp16.onnx")
    )
    print(f"fp16 onnx ~{size/1024/1024:.0f} MiB", flush=True)
    print("EXPORT_DONE", flush=True)


def cmd_quantize(args):
    import inspect

    import onnx

    t0 = time.time()
    fp16_path = os.path.join(args.work, "model.fp16.onnx")
    print("loading fp16 graph (external data) ...", flush=True)
    model = onnx.load_model(fp16_path, load_external_data=True)
    print(f"loaded  rss={rss_gb():.2f}GB  ({time.time()-t0:.0f}s)", flush=True)

    # ORT >= 1.21 renamed the module to matmul_nbits_quantizer (MatMulNBits op)
    try:
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer as Q
    except Exception:
        from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer as Q

    sig = inspect.signature(Q.__init__)
    print("quantizer:", Q.__name__, "params:", list(sig.parameters), flush=True)

    print(f"quantising MatMuls -> int4 (block={args.block}, symmetric={args.symmetric}) ...", flush=True)
    quant = Q(model, bits=4, block_size=args.block, is_symmetric=bool(args.symmetric))
    quant.process()
    print(f"quantised rss={rss_gb():.2f}GB  ({time.time()-t0:.0f}s)", flush=True)

    # ORT 1.31: the quantised graph lives inside the ONNXModel wrapper, usually as an
    # onnx_ir.Model. Pick it by actual type and verify MatMulNBits nodes exist.
    wrapper = getattr(quant, "model", None)
    wm = getattr(wrapper, "model", None)
    print("type(quant.model)", type(wrapper), "| wrapper.model", type(wm), flush=True)

    try:
        import onnx_ir as ir
    except Exception:
        ir = None

    res = None
    for obj in (wm, wrapper, model):
        if obj is None:
            continue
        if type(obj).__name__ == "ModelProto":
            res = obj
            print("selected ModelProto", flush=True)
            break
        if ir is not None and isinstance(obj, ir.Model):
            print("converting onnx_ir.Model -> ModelProto", flush=True)
            res = ir.to_proto(obj)
            break
    if res is None:
        raise SystemExit("FATAL: could not obtain the quantised model object")

    nbits = sum(1 for n in res.graph.node if n.op_type == "MatMulNBits")
    print(f"MatMulNBits nodes: {nbits} (must be > 0)", flush=True)
    if nbits == 0:
        raise SystemExit("FATAL: no MatMulNBits nodes - quantisation was not applied")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "model.onnx")
    data_file = "model.onnx.data"
    onnx.save_model(
        res, out_path, save_as_external_data=True, all_tensors_to_one_file=True,
        location=data_file, size_threshold=0, convert_attribute=False,
    )

    gsz = os.path.getsize(out_path) / (1024 ** 3)
    dsz = os.path.getsize(os.path.join(args.out, data_file)) / (1024 ** 3)
    print(f"int4 onnx {gsz:.2f} GiB + data {dsz:.2f} GiB (embed tokens stay fp16)", flush=True)
    if dsz > 3.2:
        raise SystemExit(f"FATAL: external data {dsz:.2f} GiB - that is the fp16 graph, not int4")
    print("QUANTIZE_DONE", flush=True)


def cmd_fold(args):
    """Fold Transpose(initializer) into the weights, streaming one tensor at a time.

    torch.export emits Linear as MatMul(x, Transpose(W)); ORT's int4 quantiser only
    inspects node.input[1] and skips anything that is not an initializer, so without
    this step every single MatMul is skipped and no MatMulNBits node is produced.

    Streaming matters here: the 8GB cgroup dies if we hold the 4.7G model in RAM and
    duplicate it while transposing, and the 10G disk dies if we write a second copy
    before deleting the first.
    """
    import numpy as np

    import onnx

    t0 = time.time()
    dirpath = args.work
    fp16_path = os.path.join(dirpath, "model.fp16.onnx")
    print("loading graph (no external data) ...", flush=True)
    m = onnx.load_model(fp16_path, load_external_data=False)
    g = m.graph
    print(f"loaded rss={rss_gb():.2f}GB nodes={len(g.node)} init={len(g.initializer)} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # onnx data type -> numpy
    NP = {1: np.float32, 10: np.float16, 7: np.int64, 6: np.int32, 11: np.float64}

    def ext_file(init):
        for e in init.external_data:
            if e.key == "location":
                return e.value
        return None

    inits = {i.name: i for i in g.initializer}
    consumers = {}
    for n in g.node:
        for inp in n.input:
            consumers[inp] = consumers.get(inp, 0) + 1

    folded = skipped = 0
    for n in list(g.node):
        if n.op_type != "Transpose" or len(n.input) != 1:
            continue
        src_name = n.input[0]
        if src_name not in inits or consumers.get(src_name, 0) != 1:
            skipped += 1
            continue
        src = inits[src_name]
        fn = ext_file(src)
        if fn is None or len(src.dims) != 2 or src.data_type not in NP:
            skipped += 1
            continue
        path = os.path.join(dirpath, fn)
        if not os.path.isfile(path):
            skipped += 1
            continue

        old_dims = list(src.dims)                    # [out, in]
        arr = np.fromfile(path, dtype=NP[src.data_type]).reshape(old_dims)
        arr_t = np.ascontiguousarray(arr.T)          # -> [in, out]
        new_fn = fn + ".folded"
        arr_t.tofile(os.path.join(dirpath, new_fn))
        del arr, arr_t

        for e in src.external_data:
            if e.key == "location":
                e.value = new_fn
        src.ClearField("dims")
        src.dims.extend([int(old_dims[1]), int(old_dims[0])])

        os.remove(path)
        out_name = n.output[0]
        for node in g.node:
            for k, name in enumerate(node.input):
                if name == out_name:
                    node.input[k] = src_name
        g.node.remove(n)
        folded += 1
        if folded % 50 == 0:
            print(f"  folded {folded} ({time.time()-t0:.0f}s, rss={rss_gb():.2f}GB)", flush=True)

    print(f"folded={folded} skipped={skipped} rss={rss_gb():.2f}GB", flush=True)
    mm = [x for x in g.node if x.op_type == "MatMul"]
    init_names = {i.name for i in g.initializer}
    ok = sum(1 for x in mm if len(x.input) > 1 and x.input[1] in init_names)
    print(f"MatMul with const weight: {ok}/{len(mm)}", flush=True)

    onnx.save_model(
        m, fp16_path, save_as_external_data=True, all_tensors_to_one_file=False,
        size_threshold=0, convert_attribute=False,
    )
    tot = sum(os.path.getsize(os.path.join(dirpath, f)) for f in os.listdir(dirpath))
    print(f"fp16 graph rewritten: {tot/1024/1024/1024:.2f} GiB ({time.time()-t0:.0f}s)", flush=True)
    print("FOLD_DONE", flush=True)


def cmd_package(args):
    keep = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]
    os.makedirs(args.out, exist_ok=True)
    for f in keep:
        src = os.path.join(args.model, f)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(args.out, f))
    print("packaged:", sorted(os.listdir(args.out)), flush=True)
    print("PACKAGE_DONE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["export", "fold", "quantize", "package"])
    p.add_argument("--model", default="/home/daytona/merged")
    p.add_argument("--work", default="/home/daytona/onnx_work")
    p.add_argument("--out", default="/home/daytona/onnx_out")
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--symmetric", type=int, default=1)
    a = p.parse_args()
    {"export": cmd_export, "fold": cmd_fold, "quantize": cmd_quantize,
     "package": cmd_package}[a.mode](a)
