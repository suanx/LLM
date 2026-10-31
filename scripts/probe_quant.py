import inspect

import onnxruntime

print("ort", onnxruntime.__version__)
try:
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer as Q

    print("MatMulNBitsQuantizer OK")
    print("Q sig:", inspect.signature(Q.__init__))
    print("Q methods:", [m for m in dir(Q) if not m.startswith("_")])
except Exception as e:
    print("nbits err:", e)

import pkgutil
import onnxruntime.quantization.matmul_nbits_quantizer as m

print("module names:", [n for n in dir(m) if not n.startswith("_")])
for name in ("DefaultWeightOnlyQuantConfig", "WeightOnlyQuantConfig"):
    if hasattr(m, name):
        print(name, "sig:", inspect.signature(getattr(m, name).__init__))

try:
    from onnxruntime.quantization.quant_utils import QuantType

    print("QuantType attrs:", [a for a in dir(QuantType) if not a.startswith("_")][:12])
except Exception as e:
    print("qt err:", e)
print("PROBE_DONE")
