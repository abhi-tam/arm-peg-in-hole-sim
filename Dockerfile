FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git ffmpeg build-essential \
    libegl1 libgl1 libglvnd0 libgles2 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Register the NVIDIA EGL vendor so headless GPU rendering is found
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /workspace
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .
RUN pip install -e .
ENTRYPOINT ["python"]
CMD ["evaluate.py", "--help"]
