# Image DeepStream 7.0 có sẵn trên máy + Python bindings (pyds 1.1.11 = bản dành cho DeepStream 7.0).
# Build 1 lần: docker build -t aipark-ds:7.0 people_count/ds
FROM nvcr.io/nvidia/deepstream:7.0-triton-multiarch
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-gi python3-dev python3-gst-1.0 python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir \
    https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.11/pyds-1.1.11-py3-none-linux_x86_64.whl
