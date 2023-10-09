import openvino as ov
from os import path
import onnx
import sys

onnx_path = sys.argv[1]
ov_model = ov.convert_model(onnx_path)
ov.save_model(ov_model, path.splitext(path.basename(onnx_path))[0] + "-ov.xml")