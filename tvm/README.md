Description of the TVM workflow

Compiling
```sh
tvmc compile --target "llvm" --input-shapes "data:[1,3,{Height},{Width}]" --output model-tvm.tar model.onnx
```

Running
```sh
tvmc run --inputs {input.npz} --output {output.npz} model-tvm.tar
```

Tuning
```sh
tvmc tune --target "llvm" --output model-autotuner-records.log model.onnx
```