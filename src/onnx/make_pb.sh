#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/lib/python3.6/dist-packages/trtpy/trt8cuda112cudnn8/lib64
protoc=/usr/local/lib/python3.6/dist-packages/trtpy/trt8cuda112cudnn8/bin/protoc

cd onnx

echo Create directory "pbout"
rm -rf pbout
mkdir -p pbout

$protoc onnx-ml.proto --cpp_out=pbout
$protoc onnx-operators-ml.proto --cpp_out=pbout

echo Copy pbout/onnx-ml.pb.cc to ../src/tensorRT/onnx/onnx-ml.pb.cpp
cp pbout/onnx-ml.pb.cc           ../src/tensorRT/onnx/onnx-ml.pb.cpp

echo Copy pbout/onnx-operators-ml.pb.cc to ../src/tensorRT/onnx/onnx-operators-ml.pb.cpp
cp pbout/onnx-operators-ml.pb.cc ../src/tensorRT/onnx/onnx-operators-ml.pb.cpp

echo Copy pbout/onnx-ml.pb.h to ../src/tensorRT/onnx/onnx-ml.pb.h
cp pbout/onnx-ml.pb.h           ../src/tensorRT/onnx/onnx-ml.pb.h

echo Copy pbout/onnx-operators-ml.pb.h to ../src/tensorRT/onnx/onnx-operators-ml.pb.h
cp pbout/onnx-operators-ml.pb.h ../src/tensorRT/onnx/onnx-operators-ml.pb.h

echo Remove directory "pbout"
rm -rf pbout