import inspect

import onnx
import onnx_ir as ir

print("onnx", onnx.__version__)
print("ir funcs:", [n for n in dir(ir) if not n.startswith("_")][:40])
for name in ("to_proto", "from_proto", "save", "load"):
    if hasattr(ir, name):
        try:
            print(f"ir.{name} sig:", inspect.signature(getattr(ir, name)))
        except Exception as e:
            print(f"ir.{name} sig err {e}")

import onnx_ir.serde as serde

print("serde:", [n for n in dir(serde) if not n.startswith("_")][:20])
if hasattr(serde, "serialize"):
    print("serialize sig:", inspect.signature(serde.serialize))
print("PROBE_DONE")
