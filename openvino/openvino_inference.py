import openvino as ov
import sys
import numpy as np
import time

core = ov.Core()
for device in core.available_devices:
    print(core.get_property(device, "FULL_DEVICE_NAME"))

ir_path = sys.argv[1]
model_ir = core.read_model(model=ir_path)
compiled_model_ir = core.compile_model(model=model_ir, device_name="CPU")

input_image = np.random.randn(1,3,120,160)
output_layer_ir = compiled_model_ir.output(0)

def evaluate_model():
    input_image = np.random.randn(1,3,120,160)
    res_ir = compiled_model_ir([input_image])[output_layer_ir]
    return res_ir

# warmup:
for _ in range(10):
    evaluate_model()

# benchmark
times = []
for _ in range(10000):
    now = time.time()
    evaluate_model()
    times.append(time.time() - now)

times = np.array(times)

print(f"""
min/max: [{1000 * np.min(times)}, {1000 * np.max(times)}]
mean: {1000 * np.mean(times):.2f}ms
median: {1000 * np.median(times):.2f}ms
stddev: {1000 * np.std(times):.2f}ms
""")

res_ir = compiled_model_ir([input_image])[output_layer_ir]

print(res_ir.shape)
