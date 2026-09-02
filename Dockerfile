# TRELLIS.2 (image -> textured GLB) as a frozen, digest-pinnable environment.
# This image is the lockfile: Microsoft's setup.sh pins nothing, so a fresh install drifts
# (opencv 5 broke EXR, transformers 5.16 moved DINOv3's layers, pillow-simd broke WebP).
# Everything below reproduces the recipe that generated on an A40 on 2026-09-01.
# Contains no credentials and nothing site-specific; weights are fetched at run time.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel@sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56

ARG DEBIAN_FRONTEND=noninteractive
ARG TRELLIS_COMMIT=75fbf0183001ed9876c8dbb35de6b68552ee08bd
# 8.6 = A40/A6000/A10; 8.9 = L4/L40S/RTX 40xx; both so the same image runs on the dev box.
ARG TORCH_CUDA_ARCH_LIST="8.6;8.9"
ARG MAX_JOBS=12

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates curl git libjpeg-dev \
    && rm -rf /var/lib/apt/lists/* \
    && printf '#!/bin/sh\nexec "$@"\n' > /usr/local/bin/sudo && chmod 0755 /usr/local/bin/sudo

RUN git clone --recurse-submodules https://github.com/microsoft/TRELLIS.2.git /opt/trellis2 \
    && git -C /opt/trellis2 checkout --detach "${TRELLIS_COMMIT}" \
    && git -C /opt/trellis2 submodule update --init --recursive

COPY dinov3_patch.py /opt/foundry/dinov3_patch.py
RUN python /opt/foundry/dinov3_patch.py /opt/trellis2/trellis2/modules/image_feature_extractor.py

ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PIP_NO_CACHE_DIR=1

# setup.sh only uses nvidia-smi to detect the platform; the build host has no GPU, so shim it
# for the install and remove the shim afterwards (it would shadow the driver's real binary).
WORKDIR /opt/trellis2
RUN printf '#!/bin/sh\nexit 0\n' > /usr/local/bin/nvidia-smi && chmod 0755 /usr/local/bin/nvidia-smi \
    && bash -lc 'source /opt/conda/etc/profile.d/conda.sh; export MAX_JOBS='"${MAX_JOBS}"'; cd /opt/trellis2 && source ./setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm' \
    && rm -f /usr/local/bin/nvidia-smi \
    && rm -rf /tmp/extensions /root/.cache/pip

# Drift fixes proven on the A40 (see docs/runbooks/foundry/CLAIMS.md):
#  - opencv-python-headless 5.0.0 cannot decode the EXR HDRI; 4.x can.
#  - pillow-simd's _webp lacks HAVE_WEBPANIM and breaks the GLB texture writer; one clean Pillow.
RUN pip install --force-reinstall "opencv-python-headless<5" \
    && pip uninstall -y pillow-simd \
    && pip install --force-reinstall pillow \
    && rm -rf /root/.cache/pip

COPY image_to_glb.py /opt/foundry/image_to_glb.py

ENV PYTHONPATH=/opt/trellis2 \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    HF_HOME=/workspace/hf \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PYTHONUNBUFFERED=1

# Build-time proof (no GPU needed): every extension imports, WebP writes, EXR decodes.
RUN python -c "import torch, flash_attn, nvdiffrast, o_voxel, cumesh, flex_gemm, trellis2; print({'torch': torch.__version__, 'cuda': torch.version.cuda, 'flash_attn': flash_attn.__version__})" \
    && python -c "from PIL import Image; import io; Image.new('RGB',(4,4)).save(io.BytesIO(), format='WEBP'); print('PILLOW_WEBP=OK')" \
    && python -c "import cv2; assert cv2.__version__.startswith('4.'), cv2.__version__; import numpy; im = cv2.imread('/opt/trellis2/assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED); assert im is not None, 'EXR decode failed'; print('EXR_DECODE=OK', cv2.__version__, im.shape)" \
    && python -c "import transformers, PIL; print({'transformers': transformers.__version__, 'pillow': PIL.__version__})" \
    && pip freeze > /opt/foundry/PIP-FREEZE.txt

# Best effort: pre-compile nvdiffrast's CUDA plugin so the first run on a GPU skips the JIT.
RUN python -c "import nvdiffrast.torch.ops as ops; ops._get_plugin(); print('NVDIFFRAST_PLUGIN=PREBUILT')" || echo "NVDIFFRAST_PLUGIN=JIT_AT_RUNTIME"

LABEL org.opencontainers.image.title="trellis2-env" \
      org.opencontainers.image.description="Microsoft TRELLIS.2 image-to-3D, pinned and working (torch 2.6.0+cu124, flash-attn 2.7.3, opencv 4, DINOv3 patched for transformers 5.x)" \
      org.opencontainers.image.revision="${TRELLIS_COMMIT}" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /workspace
CMD ["sleep", "infinity"]
