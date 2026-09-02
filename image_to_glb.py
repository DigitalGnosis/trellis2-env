"""Image -> textured GLB with TRELLIS.2. Baked into the foundry image.

usage: image_to_glb.py INPUT_IMAGE OUT_DIR [--no-video] [--texture-size N] [--decimate N] [--pipeline-type 1536_cascade] [--seed N]
Writes OUT_DIR/model.glb (PNG textures), OUT_DIR/turntable.mp4 (unless --no-video),
OUT_DIR/marks.json (timings, peak VRAM, mesh counts).
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

parser = argparse.ArgumentParser()
parser.add_argument("input_image")
parser.add_argument("out_dir")
parser.add_argument("--no-video", action="store_true")
parser.add_argument("--texture-size", type=int, default=4096)
parser.add_argument("--decimate", type=int, default=1000000)
parser.add_argument("--hdri", default="/opt/trellis2/assets/hdri/forest.exr")
parser.add_argument("--model", default="microsoft/TRELLIS.2-4B")
parser.add_argument("--pipeline-type", default=None, choices=[None, "512", "1024", "1024_cascade", "1536_cascade"],
                    help="TRELLIS.2 resolution mode (default: the model's own default, 1024_cascade for TRELLIS.2-4B)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

import cv2, imageio, torch  # noqa: E402
from PIL import Image  # noqa: E402
from trellis2.pipelines import Trellis2ImageTo3DPipeline  # noqa: E402
from trellis2.renderers import EnvMap  # noqa: E402
from trellis2.utils import render_utils  # noqa: E402
import o_voxel  # noqa: E402

os.makedirs(args.out_dir, exist_ok=True)
marks = {"input": os.path.basename(args.input_image), "cv2": cv2.__version__, "torch": torch.__version__}
t = time.time()
envmap = None
if not args.no_video:
    hdri = cv2.imread(args.hdri, cv2.IMREAD_UNCHANGED)
    if hdri is None:
        marks["video"] = f"skipped: cv2 could not decode {args.hdri}"
    else:
        envmap = EnvMap(torch.tensor(cv2.cvtColor(hdri, cv2.COLOR_BGR2RGB), dtype=torch.float32, device="cuda"))
pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
pipeline.cuda()
marks["load_s"] = round(time.time() - t, 1)
t = time.time()
image = Image.open(args.input_image)
marks["input_size"] = list(image.size)
marks["input_mode"] = image.mode
torch.cuda.reset_peak_memory_stats()
mesh = pipeline.run(image, seed=args.seed, pipeline_type=args.pipeline_type)[0]
marks["pipeline_type"] = args.pipeline_type or getattr(pipeline, "default_pipeline_type", "default")
marks["seed"] = args.seed
marks["run_s"] = round(time.time() - t, 1)
marks["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
t = time.time()
mesh.simplify(16777216)
if envmap is not None:
    try:
        video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
        imageio.mimsave(os.path.join(args.out_dir, "turntable.mp4"), video, fps=15)
        marks["video_s"] = round(time.time() - t, 1)
    except Exception as exc:  # the GLB is the deliverable; the video is a bonus
        marks["video"] = f"failed: {type(exc).__name__}: {exc}"[:300]
t = time.time()
glb = o_voxel.postprocess.to_glb(
    vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
    attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target=args.decimate, texture_size=args.texture_size,
    remesh=True, remesh_band=1, remesh_project=0, verbose=True,
)
glb.export(os.path.join(args.out_dir, "model.glb"), extension_webp=False)  # PNG textures
marks["glb_s"] = round(time.time() - t, 1)
marks["vertices"] = int(mesh.vertices.shape[0])
marks["faces"] = int(mesh.faces.shape[0])
marks["glb_bytes"] = os.path.getsize(os.path.join(args.out_dir, "model.glb"))
json.dump(marks, open(os.path.join(args.out_dir, "marks.json"), "w"), indent=1)
print(marks)
