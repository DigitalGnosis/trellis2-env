# trellis2-env

[Microsoft TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (image → textured 3D mesh), frozen into a
container image that actually runs. Upstream's `setup.sh` pins nothing, and a fresh install in
September 2026 drifted three ways; this image is the lockfile.

What is fixed relative to a fresh `setup.sh`:

- `opencv-python-headless` 5.0.0 cannot decode the EXR HDRI used for the turntable video; pinned `< 5`.
- `transformers` 5.x moved DINOv3's layers, so `image_feature_extractor.py` crashed with
  `'DINOv3ViTModel' object has no attribute 'layer'`; `dinov3_patch.py` finds the layer list instead of
  naming its path (works on 4.x and 5.x layouts).
- `pillow-simd` installed over Pillow breaks the GLB texture writer (`PIL._webp` lacks `HAVE_WEBPANIM`);
  one clean Pillow, textures written as PNG.

Base: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` (digest pinned). TRELLIS.2 at commit
`75fbf0183001ed9876c8dbb35de6b68552ee08bd`. CUDA kernels built for sm_86 and sm_89.
Nothing site-specific and no credentials inside; model weights are downloaded at run time.

## Run

```
docker run --gpus all -e HF_TOKEN=... -v $PWD/out:/out ghcr.io/digitalgnosis/trellis2-env:latest \
  python /opt/foundry/image_to_glb.py /opt/trellis2/assets/example_image/T.png /out
```

`HF_TOKEN` must belong to an account that accepted the gates on
`facebook/dinov3-vitl16-pretrain-lvd1689m` and `briaai/RMBG-2.0` (the latter is CC BY-NC 4.0).
Outputs: `model.glb` (PNG textures), `turntable.mp4`, `marks.json` (timings, peak VRAM, mesh counts).
Measured on an A40: pipeline load ~150 s, generation ~180 s, peak VRAM 7.1 GB, 4K texture bake ~105 s.

Pin by digest, not by tag; the digest is printed in each build's job summary.
