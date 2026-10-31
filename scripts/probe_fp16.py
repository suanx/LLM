import os

import onnx

p = "/home/daytona/onnx_work/model.fp16.onnx"
m = onnx.load_model(p, load_external_data=True)
g = m.graph
print("ir_version", m.ir_version, "opset", [(o.domain, o.version) for o in m.opset_import])
print("initializers", len(g.initializer), "nodes", len(g.node), "inputs", len(g.input))
print("graph inputs:", [i.name for i in g.input][:6])

init_names = {i.name for i in g.initializer}
matmuls = [n for n in g.node if n.op_type == "MatMul"]
print("MatMul nodes:", len(matmuls))
for n in matmuls[:6]:
    print("  node", n.name, "inputs", list(n.input),
          "-> input[1] is initializer:", n.input[1] in init_names if len(n.input) > 1 else "n/a")

# do any initializers actually carry raw bytes?
carry = sum(1 for i in g.initializer if len(i.raw_data) > 0)
print("initializers with raw_data:", carry)
if g.initializer:
    i0 = g.initializer[0]
    print("sample initializer:", i0.name, "dims", list(i0.dims), "raw", len(i0.raw_data),
          "data_location", i0.data_location)

from onnxruntime.quantization.onnx_model import ONNXModel

om = ONNXModel(m)
got = om.get_initializer(matmuls[0].input[1]) if matmuls else None
print("ONNXModel.get_initializer ->", type(got), "dims", list(got.dims) if got is not None else None)
print("PROBE_DONE")
