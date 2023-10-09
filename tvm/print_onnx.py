import onnx
import sys
model = onnx.load(sys.argv[1])
print(model.graph)