# Image engine overview: DeepStream 7.0 + Python bindings + thư viện parser YOLO build NGAY TRONG image.
# Build 1 lần (từ thư mục AI_park): docker build -t aipark-overview:7.0 people_count
# Parser có code CUDA -> build ở đây để khớp CUDA / DeepStream của chính image này (KIEN_TRUC mục 10),
# không nhặt file .so build sẵn từ chỗ khác.
FROM nvcr.io/nvidia/deepstream:7.0-triton-multiarch
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-gi python3-dev python3-gst-1.0 python3-pip git \
    && rm -rf /var/lib/apt/lists/*
# pyds 1.1.11 = bản dành cho DeepStream 7.0
RUN pip3 install --no-cache-dir \
        https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.11/pyds-1.1.11-py3-none-linux_x86_64.whl \
        pyyaml python-dotenv
# DeepStream-Yolo: cùng commit đang dùng trên robo (~/my_deepstream_test/DeepStream-Yolo); CUDA 12.2 = CUDA của DeepStream 7.0
RUN git clone https://github.com/marcoslucianops/DeepStream-Yolo.git /opt/DeepStream-Yolo \
    && cd /opt/DeepStream-Yolo && git checkout 2894bab \
    && CUDA_VER=12.2 make -C nvdsinfer_custom_impl_Yolo \
    && mkdir -p /opt/aipark && cp nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so /opt/aipark/
# Entrypoint gốc của image (deepstream-7.0/entrypoint.sh) chạy python làm TIẾN TRÌNH CON của bash và tách lại tham số
# theo dấu cách -> lệnh dừng (docker stop = SIGTERM, Ctrl+C = SIGINT) dừng ở bash, không tới python
# (đo 2026-10-19 bằng docker top: docker-init -> bash entrypoint.sh -> python3). Ở đây gọi thẳng phần khởi động của
# NVIDIA bằng exec: python THAY THẾ shell, nhận tín hiệu trực tiếp. Chạy kèm docker run --init.
RUN printf '%s\n' '#!/bin/bash' \
        'if [ -x /opt/nvidia/nvidia_entrypoint.sh ]; then exec /opt/nvidia/nvidia_entrypoint.sh "$@"; fi' \
        'exec "$@"' > /opt/aipark/entrypoint.sh \
    && chmod +x /opt/aipark/entrypoint.sh
ENTRYPOINT ["/opt/aipark/entrypoint.sh"]