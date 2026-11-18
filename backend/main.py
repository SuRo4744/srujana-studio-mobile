"""
Srujana - Problem 3 backend: video -> sharp frames -> Gaussian splat -> web-viewable 3D model.

Pipeline
  1. Laplacian-variance blur filter, keep the 2-3 sharpest frames per second
  2. Camera poses (COLMAP) + 3D Gaussian Splatting training  (via nerfstudio "splatfacto")
  3. Export the trained splat as a .ply that the frontend renders in the browser

Setup (needs an NVIDIA GPU + CUDA for step 2):
  pip install fastapi uvicorn python-multipart opencv-python numpy
  pip install nerfstudio            # also install COLMAP and make sure `colmap` is on PATH
  (no GPU?) pip install pycolmap    # CPU-only point cloud mode, runs on a normal laptop
  python backend.py                 # serves on http://localhost:8000
"""
import base64
import hmac
import json
import re
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from exporters import export_obj, export_blend, export_mesh_format
from surface import clean_occupancy, export_surface
from quality import analyze_video

JOBS_DIR = Path(__file__).parent / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

# ---- tunables -------------------------------------------------------------
FRAMES_PER_SEC = 3        # keep at most this many sharp frames per second
BLUR_THRESHOLD = 100.0    # variance of Laplacian below this = blurry (tune per camera)
MAX_WIDTH = 1280          # frames are downscaled before scoring and saving
MIN_FRAMES = 20           # fewer than this and COLMAP/splatting will not work well
SCORE_STRIDE = 2          # score every Nth video frame (skipping the rest is much faster)
TT_GRID = int(os.environ.get("TT_GRID", "160"))        # turntable mode: voxels across (higher = finer, slower)
TT_VIEWS = int(os.environ.get("TT_VIEWS", "36"))       # turntable mode: number of viewing angles used
TT_WIDTH = int(os.environ.get("TT_WIDTH", "640"))   # turntable mode: frame width used (higher = sharper colours)
BG_THRESHOLD = os.environ.get("BG_THRESHOLD", "auto")  # "auto" or a number: how different from the backdrop counts as "object"
ROT_DIR = os.environ.get("ROT_DIR", "auto")            # auto | 1 | -1  (spin direction, if auto-detect is wrong)
SFM_FRAMES = int(os.environ.get("SFM_FRAMES", "24"))  # CPU mode: max frames used (fewer = faster)
SFM_WIDTH = int(os.environ.get("SFM_WIDTH", "720"))   # CPU mode: frame width used (smaller = faster)
TRAIN_ITERATIONS = 15000  # lower (e.g. 7000) for a faster demo, higher for quality

# Capture styles
#   turntable : camera still, object spins one full turn on a plain backdrop. No GPU, a few seconds. (fastest)
#   orbit     : camera walks around a still object. cpu = point cloud (minutes), full = Gaussian splat (GPU).
# MODE: "auto" (default) picks the best option available on this machine:
#         GPU tools (nerfstudio + COLMAP)  -> "full"  Gaussian splat (needs NVIDIA GPU)
#         else `pip install pycolmap`      -> "cpu"   colored point cloud, no GPU needed
#         else                             -> "frames_only"  just the sharp frames
#       You can also force one: MODE=full | cpu | frames_only
# DEMO_PLY: optional path to a model built earlier; shown (and labelled as pre-built)
#           when the 3D step is skipped.
MODE = os.environ.get("MODE", "auto")
DEMO_PLY = os.environ.get("DEMO_PLY", "")

app = FastAPI(title="Srujana 3D backend")
CLOUD_DEPLOYMENT = os.environ.get("CLOUD_DEPLOYMENT") == "1"
if not CLOUD_DEPLOYMENT:
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_cloud_job_lock = threading.Lock()


@app.middleware("http")
async def cloud_password_gate(request: Request, call_next):
    """Protect uploads, history and model downloads, as well as static HTML.

    This is a single-user studio, not multi-user identity management.
    Uses HTTP Basic exclusively behind the HTTPS URL supplied by Render.
    """
    if not CLOUD_DEPLOYMENT or request.url.path == "/health":
        return await call_next(request)
    password = os.environ.get("STUDIO_PASSWORD", "")
    if not password:
        return JSONResponse(status_code=503, content={"detail": "Set STUDIO_PASSWORD in the Render environment before using this app."})
    header = request.headers.get("authorization", "")
    expected = base64.b64encode(("studio:" + password).encode()).decode()
    supplied = header[6:].strip() if header.lower().startswith("basic ") else ""
    if not hmac.compare_digest(supplied, expected):
        return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Srujana Studio"', "Cache-Control": "no-store"})
    return await call_next(request)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def client_config():
    return {"cloud": CLOUD_DEPLOYMENT, "max_upload_mb": int(os.environ.get("MAX_UPLOAD_MB", "600")),
            "turntable_only": CLOUD_DEPLOYMENT, "blender_available": not CLOUD_DEPLOYMENT and bool(shutil.which("blender"))}



# ---- job state ------------------------------------------------------------
def job_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", job_id):
        raise HTTPException(400, "Invalid job ID")
    return JOBS_DIR / job_id


def set_status(job_id: str, stage: str, message: str = "", **extra):
    data = {"id": job_id, "stage": stage, "message": message, "updated": time.time(), **extra}
    # Atomic replacement prevents polling clients from reading half a JSON file.
    path = job_dir(job_id) / "status.json"
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(data))
    temp_path.replace(path)


def get_status(job_id: str) -> dict:
    path = job_dir(job_id) / "status.json"
    if not path.exists():
        raise HTTPException(404, "Unknown job")
    return json.loads(path.read_text())


# ---- step 1: blur filtering and frame selection ----------------------------
def blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian: high = sharp edges, low = blurry."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_sharp_frames(video_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open the video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    min_gap = max(1, int(fps / (FRAMES_PER_SEC * 2)))  # keep chosen frames visually distinct

    kept, total, discarded_blurry = 0, 0, 0
    bucket = []        # (score, frame_idx, image) for the current second
    current_sec = 0

    def flush():
        nonlocal kept, discarded_blurry
        if not bucket:
            return
        bucket.sort(key=lambda b: b[0], reverse=True)
        chosen = []
        for score, idx, img in bucket:
            if len(chosen) >= FRAMES_PER_SEC:
                break
            if score < BLUR_THRESHOLD:
                continue
            if any(abs(idx - c[1]) < min_gap for c in chosen):
                continue
            chosen.append((score, idx, img))
        discarded_blurry += len(bucket) - len(chosen)
        for score, idx, img in sorted(chosen, key=lambda c: c[1]):
            cv2.imwrite(str(out_dir / f"frame_{kept:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            kept += 1
        bucket.clear()

    while True:
        if total % SCORE_STRIDE != 0:      # skip decoding work for frames we will not score
            if not cap.grab():
                break
            total += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        sec = int(total / fps)
        if sec != current_sec:
            flush()
            current_sec = sec
        h, w = frame.shape[:2]
        if w > MAX_WIDTH:
            frame = cv2.resize(frame, (MAX_WIDTH, int(h * MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bucket.append((blur_score(gray), total, frame))
        total += 1
    flush()
    cap.release()

    return {"fps": round(fps, 2), "total_frames": total, "kept_frames": kept,
            "discarded_frames": total - kept, "blurry_discarded": discarded_blurry}


# ---- step 2 + 3: Gaussian splatting and export -----------------------------
def run_cmd(cmd: list, log_file: Path):
    with open(log_file, "a") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = "\n".join(log_file.read_text().splitlines()[-15:])
        raise RuntimeError(f"`{cmd[0]}` failed:\n{tail}")


def build_splat(frames_dir: Path, work: Path, log_file: Path) -> Path:
    for tool in ("ns-process-data", "ns-train", "ns-export", "colmap"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"`{tool}` not found. Install nerfstudio and COLMAP (see top of backend.py).")

    processed = work / "processed"
    train_out = work / "train"
    export_dir = work / "export"

    # camera poses from the sharp frames (COLMAP structure-from-motion)
    run_cmd(["ns-process-data", "images", "--data", str(frames_dir),
             "--output-dir", str(processed)], log_file)

    # train 3D Gaussian Splatting
    run_cmd(["ns-train", "splatfacto", "--data", str(processed),
             "--output-dir", str(train_out),
             "--max-num-iterations", str(TRAIN_ITERATIONS),
             "--vis", "tensorboard",
             "--viewer.quit-on-train-completion", "True"], log_file)

    configs = sorted(train_out.rglob("config.yml"), key=lambda p: p.stat().st_mtime)
    if not configs:
        raise RuntimeError("Training finished but no config.yml was found")

    # export the Gaussians as a .ply file the browser viewer can open
    run_cmd(["ns-export", "gaussian-splat", "--load-config", str(configs[-1]),
             "--output-dir", str(export_dir)], log_file)

    plys = sorted(export_dir.glob("*.ply"))
    if not plys:
        raise RuntimeError("Export finished but no .ply was produced")
    return plys[0]


def pycolmap_available() -> bool:
    try:
        import pycolmap  # noqa: F401
        return True
    except Exception:
        return False


def build_pointcloud(job_id: str, frames_dir: Path, work: Path, stats: dict):
    """CPU-only 3D, tuned for speed: few small frames -> SIFT -> matching -> SfM -> colored points."""
    import pycolmap

    t0 = time.time()
    files = sorted(frames_dir.glob("*.jpg"))
    if len(files) > SFM_FRAMES:  # spread the chosen frames evenly around the orbit
        idx = sorted(set(np.linspace(0, len(files) - 1, SFM_FRAMES).round().astype(int)))
        files = [files[i] for i in idx]

    small = work / "sfm_frames"
    if small.exists():
        shutil.rmtree(small)
    small.mkdir()
    for f in files:
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        if w > SFM_WIDTH:
            img = cv2.resize(img, (SFM_WIDTH, int(h * SFM_WIDTH / w)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(small / f.name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    db = work / "colmap.db"
    sparse = work / "sparse"
    sparse.mkdir(exist_ok=True)
    if db.exists():
        db.unlink()

    set_status(job_id, "reconstructing", f"1/3 Finding features in {len(files)} frames", stats=stats)
    pycolmap.extract_features(str(db), str(small))
    set_status(job_id, "reconstructing", "2/3 Matching frames", stats=stats)
    pycolmap.match_exhaustive(str(db))
    set_status(job_id, "reconstructing", "3/3 Recovering camera positions and 3D points", stats=stats)
    maps = pycolmap.incremental_mapping(str(db), str(small), str(sparse))
    if not maps:
        raise RuntimeError("Could not recover camera positions. Film more slowly, in better light, "
                           "on a textured (not plain or shiny) surface, and circle the whole object.")
    best = max(maps.values(), key=lambda r: r.num_reg_images())
    ply = work / "pointcloud.ply"
    best.export_PLY(str(ply))
    return ply, best.num_reg_images(), len(best.points3D), round(time.time() - t0)


# ---- fast no-GPU method: turntable + silhouette carving (visual hull) ----------------
def _read_small_frames(video_path: Path, max_frames: int = 200):
    max_frames = min(max_frames, int(os.environ.get("TT_MAX_FRAMES", "200")))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open the video file")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    stride = max(1, n // max_frames) if n else 2
    frames, total = [], 0
    while True:
        if total % stride:
            if not cap.grab():
                break
            total += 1
            continue
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        if w > TT_WIDTH:
            f = cv2.resize(f, (TT_WIDTH, int(h * TT_WIDTH / w)), interpolation=cv2.INTER_AREA)
        frames.append(f)
        total += 1
    cap.release()
    return frames, fps, total, stride


def _rotation_angles(frames):
    """Estimate each frame's turntable angle from how much the picture changes (handles uneven spin)."""
    grays = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32) for f in frames]
    diff = np.array([np.abs(grays[i + 1] - grays[i]).mean() for i in range(len(grays) - 1)])
    # a median filter removes one-off spikes (a blurry frame, a flicker) so they do not count as rotation
    padded = np.pad(diff, 3, mode="edge")
    diff = np.median(np.lib.stride_tricks.sliding_window_view(padded, 7), axis=1)
    diff = np.convolve(diff, np.ones(3) / 3, mode="same")
    peak = float(np.percentile(diff, 90))
    floor = float(np.percentile(diff, 10))
    if peak < 0.05 or peak < 2.0 * floor:
        raise RuntimeError("No rotation detected. Keep the camera still and spin the object one full turn.")
    active = np.where(diff > max(0.25 * peak, 1.5 * floor))[0]
    a, b = int(active[0]), int(active[-1]) + 1

    # if the clip holds several turns (e.g. a looping stock video), keep only the first full turn:
    # the picture comes back to the first frame's look when one turn is done
    small = [cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames[a:b + 1]]
    dist = np.array([np.abs(g - small[0]).mean() for g in small])
    top = float(dist.max())
    if top > 1.0:
        far = np.where(dist > 0.6 * top)[0]
        if len(far):
            for i in range(int(far[0]) + 1, len(dist) - 1):
                if dist[i] < 0.25 * top and dist[i] <= dist[i - 1] and dist[i] <= dist[i + 1]:
                    b = a + i
                    break
    cum = np.concatenate([[0.0], np.cumsum(diff)])
    seg = cum[a:b + 1] - cum[a]
    return a, b, 360.0 * seg / seg[-1]


def _odd(v) -> int:
    v = max(3, int(v))
    return v if v % 2 else v + 1


def _ellipse(k: int):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _segment(frame, bg, thr, k_open=3, k_close=5):
    """Object mask = pixels that differ from the backdrop colour, cleaned up; largest blob only."""
    d = np.abs(frame.astype(np.int16) - bg[None, None, :].astype(np.int16)).max(axis=2)
    m = (d > thr).astype(np.uint8)
    if k_open > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _ellipse(k_open))     # removes thin wisps such as steam
    if k_close > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ellipse(k_close))   # fills dark patches inside the object
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))               # drops watermarks and stray specks
    return lab == k


def _auto_threshold(strips, bg) -> float:
    if BG_THRESHOLD != "auto":
        return float(BG_THRESHOLD)
    d = np.abs(strips.astype(np.int16) - bg[None, :].astype(np.int16)).max(axis=1)
    noise = float(np.percentile(d, 98))
    if float(bg.mean()) < 40:                                # dark backdrop: dark objects need a very low cut-off
        return float(np.clip(2.0 * noise + 3.0, 3.0, 20.0))
    return float(np.clip(2.0 * noise + 10.0, 20.0, 35.0))   # bright backdrop: ignore soft shadows


def _fetch_crops(video_path: Path, frame_numbers, box, lo_size, out_width):
    """Re-read only the chosen frames from the ORIGINAL video, cropped around the object and
    resized, so colours and edges are much sharper than in the small frames."""
    cap = cv2.VideoCapture(str(video_path))
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    lw, lh = lo_size
    sx, sy = W / lw, H / lh
    x0, y0 = int(box[0] * sx), int(box[1] * sy)
    x1, y1 = int(box[2] * sx), int(box[3] * sy)
    cw, ch = max(2, x1 - x0), max(2, y1 - y0)
    ow = int(min(out_width, cw))
    oh = max(2, int(round(ch * ow / cw)))
    wanted = set(frame_numbers)
    out = {}
    n, last = 0, max(frame_numbers)
    while n <= last:
        if n in wanted:
            ok, f = cap.read()
            if not ok:
                break
            out[n] = cv2.resize(f[y0:y1, x0:x1], (ow, oh), interpolation=cv2.INTER_AREA)
        elif not cap.grab():
            break
        n += 1
    cap.release()
    return out, (ow, oh)


def _spin_direction(frames, a, b, bg, thr):
    """+1 if the front of the object moves left as the angle grows, -1 if right, 0 if unclear."""
    vals = []
    for i in np.linspace(a, b - 3, 8).astype(int):
        g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frames[i + 2], cv2.COLOR_BGR2GRAY)
        m = _segment(frames[i], bg, thr)
        if m is None:
            continue
        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        gx = cv2.Sobel(g1, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(g1, cv2.CV_32F, 0, 1)
        w = np.hypot(gx, gy) * m
        if w.sum() > 0:
            vals.append(float((flow[..., 0] * w).sum() / w.sum()))
    if not vals:
        return 0
    mean = float(np.mean(vals))
    return 1 if mean < -0.03 else (-1 if mean > 0.03 else 0)


def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    n = len(xyz)
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % n).encode()
    arr = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["r"], arr["g"], arr["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())


def build_turntable(job_id: str, video_path: Path, work: Path, quality: str = "balanced", remove_noise: bool = True):
    """Object spins one full turn in front of a still camera -> sharp view per angle ->
    silhouette carving into a voxel shape -> colored surface points. CPU only."""
    t0 = time.time()
    grid = {"fast": 96, "balanced": TT_GRID, "high": 192}.get(quality, TT_GRID)
    if CLOUD_DEPLOYMENT:
        grid = min(grid, int(os.environ.get("CLOUD_MAX_GRID", "64")))
    frames, fps, total, stride = _read_small_frames(video_path)
    if len(frames) < 30:
        raise RuntimeError("Video too short. Record 8 to 20 seconds of one full turn.")
    a, b, angles = _rotation_angles(frames)

    set_status(job_id, "filtering", "Finding the object and choosing the sharpest frame for each angle")
    H0, W0 = frames[0].shape[:2]
    step = max(1, (b - a) // 12)
    strips = np.concatenate([frames[i][:max(2, int(0.05 * H0))].reshape(-1, 3) for i in range(a, b + 1, step)])
    bg = np.median(strips, axis=0)                      # backdrop colour
    thr = _auto_threshold(strips, bg)

    # box around the object over the whole turn (lets us crop the big original frames tightly)
    bx0, by0, bx1, by1 = W0, H0, 0, 0
    for i in range(a, b + 1):
        m = _segment(frames[i], bg, thr)
        if m is None:
            continue
        x, y, w, h = cv2.boundingRect(m.astype(np.uint8))
        bx0, by0, bx1, by1 = min(bx0, x), min(by0, y), max(bx1, x + w), max(by1, y + h)
    if bx1 <= bx0 or by1 <= by0:
        raise RuntimeError("Could not find the object. Use a plain backdrop that differs in colour from the object.")
    px, py = int(0.08 * (bx1 - bx0)) + 2, int(0.08 * (by1 - by0)) + 2
    bx0, by0, bx1, by1 = max(0, bx0 - px), max(0, by0 - py), min(W0, bx1 + px), min(H0, by1 + py)

    # blur filter: variance of the Laplacian measured ONLY on the object, best frame per 10-degree bin
    lap = {i: float(cv2.Laplacian(cv2.cvtColor(frames[i][by0:by1, bx0:bx1], cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())
           for i in range(a, b + 1)}
    edges = np.linspace(0, 360, TT_VIEWS + 1)
    chosen = []
    for k in range(TT_VIEWS):
        lo, hi = edges[k], edges[k + 1]
        idx = [i for i in range(a, b + 1)
               if angles[i - a] >= lo and (angles[i - a] < hi or (k == TT_VIEWS - 1 and angles[i - a] <= hi))]
        if idx:
            chosen.append(max(idx, key=lambda i: lap[i]))
    if len(chosen) < 12:
        raise RuntimeError("Too few viewing angles. Spin the object a full turn, steadily.")

    # sharp, tightly cropped copies of just those frames from the original video
    set_status(job_id, "filtering", "Reading the chosen frames at full quality")
    crops, (ow, oh) = _fetch_crops(video_path, [i * stride for i in chosen], (bx0, by0, bx1, by1), (W0, H0), TT_WIDTH)
    views, view_angles = [], []
    for i in chosen:
        img = crops.get(i * stride)
        if img is None:  # fall back to the small frame if the original could not be read
            img = cv2.resize(frames[i][by0:by1, bx0:bx1], (ow, oh), interpolation=cv2.INTER_CUBIC)
        views.append(img)
        view_angles.append(angles[i - a])

    k_open, k_close = _odd(0.03 * ow), _odd(0.045 * ow)
    keep, masks = [], []
    for j, img in enumerate(views):
        m = _segment(img, bg, thr, k_open, k_close)
        if m is not None and m.sum() > 50:
            keep.append(j)
            masks.append(m)
    if len(keep) < 12:
        raise RuntimeError("Could not separate the object from the backdrop. Use a plain, evenly lit "
                           "background that differs in colour from the object (or set BG_THRESHOLD lower).")

    set_status(job_id, "reconstructing", "Carving the 3D shape from the silhouettes")
    if ROT_DIR in ("1", "-1"):
        sign, note = int(ROT_DIR), "spin direction set by ROT_DIR"
    else:
        sign = _spin_direction(frames, a, b, bg, thr)
        note = "spin direction auto-detected" if sign else "spin direction unclear, assumed (set ROT_DIR=-1 if colours look wrong)"
        sign = sign or 1

    H_img, W_img = masks[0].shape
    cols = [np.nonzero(m.any(axis=0))[0] for m in masks]
    rows = [np.nonzero(m.any(axis=1))[0] for m in masks]
    x0 = float(np.mean([np.nonzero(m)[1].mean() for m in masks]))      # rotation axis (image column)
    r1 = int(max(r.max() for r in rows))
    r0 = int(min(r.min() for r in rows))
    R = float(max(max(c.max() - x0, x0 - c.min()) for c in cols)) + 2.0
    height = r1 - r0 + 1
    vox = 2.0 * R / grid
    nz = int(np.ceil(height / vox))
    xs = (np.arange(grid) + 0.5) * vox - R
    zs = (np.arange(nz) + 0.5) * vox
    vi = np.floor(r1 + 0.5 - zs).astype(int)
    valid_v = (vi >= 0) & (vi < H_img)
    vi = np.clip(vi, 0, H_img - 1)

    thetas = [sign * np.radians(view_angles[j]) for j in keep]
    miss = np.zeros((nz, grid, grid), dtype=np.uint8)
    for m, th in zip(masks, thetas):
        c, s_ = np.cos(th), np.sin(th)
        ui = np.floor(x0 + 0.5 + xs[:, None] * c - xs[None, :] * s_).astype(int)
        valid_u = (ui >= 0) & (ui < W_img)
        ui = np.clip(ui, 0, W_img - 1)
        inside = m[vi][:, ui] & valid_v[:, None, None] & valid_u[None, :, :]
        miss += ~inside
    occ = miss <= max(1, int(round(0.05 * len(masks))))   # tolerate a few bad silhouettes
    if occ.sum() < 50:
        raise RuntimeError("The silhouettes did not agree. Keep the camera still and the object on the turntable axis.")
    if remove_noise:
        occ = clean_occupancy(occ)

    # surface voxels = filled voxels with an empty neighbour
    p = np.pad(occ, 1)
    core = p[1:-1, 1:-1, 1:-1]
    inner = (core & p[:-2, 1:-1, 1:-1] & p[2:, 1:-1, 1:-1] & p[1:-1, :-2, 1:-1]
             & p[1:-1, 2:, 1:-1] & p[1:-1, 1:-1, :-2] & p[1:-1, 1:-1, 2:])
    zi, xi, yi = np.nonzero(core & ~inner)
    X, Y, Z = xs[xi], xs[yi], zs[zi]

    # colour: take each point from the view where it is closest to the camera (it faces us there)
    best = np.zeros(len(X), dtype=int)
    best_t = np.full(len(X), -np.inf)
    for j, th in enumerate(thetas):
        t = X * np.sin(th) + Y * np.cos(th)
        better = t > best_t
        best[better], best_t[better] = j, t[better]
    rgb = np.zeros((len(X), 3), dtype=np.uint8)
    for j, th in enumerate(thetas):
        sel = best == j
        if not sel.any():
            continue
        u = np.clip(np.floor(x0 + 0.5 + X[sel] * np.cos(th) - Y[sel] * np.sin(th)).astype(int), 0, W_img - 1)
        v = np.clip(np.floor(r1 + 0.5 - Z[sel]).astype(int), 0, H_img - 1)
        rgb[sel] = views[keep[j]][v, u][:, ::-1]            # BGR -> RGB

    # very dark objects (like this brown cup on black) are hard to read: lift the brightness a little
    lum = float(np.median(rgb.astype(np.float32).mean(axis=1))) if len(rgb) else 0.0
    if 0 < lum < 90:
        gain = min(3.0, 90.0 / lum)
        rgb = np.clip(rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    xyz = np.stack([X, Z, -Y], axis=1) / R                  # y is up
    ply = work / "turntable.ply"
    _write_ply(ply, xyz.astype(np.float32), rgb)
    surface_info, surface_error = None, None
    try:
        # Triangle mesh is derived from the known occupied volume, not guessed
        # from a sparse cloud. Original point-cloud PLY is never overwritten.
        surface_info = export_surface(occ, xyz, rgb, R, vox, work / "surface.ply",
                                      smooth_sigma={"fast": 0.9, "balanced": 0.75, "high": 0.6}.get(quality, 0.75))
    except Exception as exc:
        surface_error = f"Surface meshing failed; original PLY is available: {exc}"
    secs = round(time.time() - t0, 1)
    stats = {"fps": round(fps, 2), "total_frames": total, "kept_frames": len(keep),
             "discarded_frames": total - len(keep), "blurry_discarded": 0,
             "points": int(len(X)), "quality_preset": quality, "grid_resolution": int(grid),
             "mesh": surface_info, "surface_error": surface_error}
    msg = f"3D model ready: {len(X)} surface points from {len(keep)} sharp views in {secs} s (turntable, no GPU; {note})"
    return ply, stats, msg


def gpu_tools_available() -> bool:
    return all(shutil.which(t) for t in ("ns-process-data", "ns-train", "ns-export", "colmap"))


def process_job(job_id: str, video_path: Path, requested: str = "auto", quality: str = "balanced", remove_noise: bool = True):
    work = job_dir(job_id)
    log_file = work / "log.txt"
    try:
        set_status(job_id, "filtering", "Checking video quality")
        report = analyze_video(video_path)
        (work / "quality.json").write_text(json.dumps(report))
        chosen_mode = requested if requested in ("turntable", "cpu", "full", "frames_only") else MODE
        if chosen_mode == "turntable":
            set_status(job_id, "filtering", "Reading the video and measuring the rotation")
            ply, stats, msg = build_turntable(job_id, video_path, work, quality, remove_noise)
            shutil.copy(ply, work / "model.ply")
            set_status(job_id, "done", msg, stats=stats, quality_report=report,
                       has_model=True, has_surface=(work / "surface.ply").exists(), model_type="voxels")
            return

        set_status(job_id, "filtering", "Removing blurry frames")
        stats = extract_sharp_frames(video_path, work / "frames")
        if stats["kept_frames"] < MIN_FRAMES:
            raise RuntimeError(
                f"Only {stats['kept_frames']} sharp frames found (need {MIN_FRAMES}+). "
                "Film more slowly, in better light, and move around the object.")

        mode = chosen_mode
        if mode == "auto":
            mode = "full" if gpu_tools_available() else ("cpu" if pycolmap_available() else "frames_only")

        if mode == "cpu":
            ply, n_img, n_pts, secs = build_pointcloud(job_id, work / "frames", work, stats)
            shutil.copy(ply, work / "model.ply")
            set_status(job_id, "done",
                       f"Point cloud ready: {n_pts} points from {n_img} frames, 3D step took {secs} s (CPU mode)",
                       stats=stats, quality_report=report, has_model=True, has_surface=False, model_type="pointcloud")
            return

        if mode == "frames_only":
            note = "Sharp frames selected. The 3D step was skipped on this machine (no GPU tools)."
            if DEMO_PLY and Path(DEMO_PLY).exists():
                shutil.copy(DEMO_PLY, work / "model.ply")
                note = "Sharp frames selected. Showing a PRE-BUILT demo model; the 3D step was not run here."
            set_status(job_id, "done", note, stats=stats, quality_report=report, has_model=(work / "model.ply").exists(), has_surface=False, model_type="pointcloud" if (work / "model.ply").exists() else "frames")
            return

        set_status(job_id, "reconstructing", "Estimating cameras and training Gaussian splats", stats=stats)
        ply = build_splat(work / "frames", work, log_file)
        shutil.copy(ply, work / "model.ply")

        set_status(job_id, "done", "Model ready", stats=stats, quality_report=report, has_model=True, has_surface=False, model_type="splat")
    except Exception as exc:  # surface any failure to the frontend
        set_status(job_id, "failed", str(exc))


# ---- API --------------------------------------------------------------------
def _cloud_purge_old_jobs(max_keep: int = 2):
    """Free web service storage is temporary; keep recent data bounded."""
    folders = sorted((p for p in JOBS_DIR.iterdir() if p.is_dir()),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in folders[max_keep:]:
        shutil.rmtree(stale, ignore_errors=True)


def _run_cloud_job(job_id, video_path, mode, quality, remove_noise):
    try:
        process_job(job_id, video_path, mode, quality, remove_noise)
    finally:
        _cloud_job_lock.release()


@app.post("/api/jobs")
async def create_job(video: UploadFile = File(...), mode: str = Form("auto"), quality: str = Form("balanced"), remove_noise: str = Form("true")):
    if quality not in ("fast", "balanced", "high"):
        raise HTTPException(400, "Invalid reconstruction quality")
    if mode not in ("turntable", "auto", "cpu", "full", "frames_only"):
        raise HTTPException(400, "Invalid capture style")
    if CLOUD_DEPLOYMENT and mode != "turntable":
        raise HTTPException(422, "Free cloud hosting supports turntable mode only. GPU and COLMAP modes are unavailable.")
    if CLOUD_DEPLOYMENT and quality == "high":
        raise HTTPException(422, "High quality exceeds the free cloud memory budget; choose Fast or Balanced.")
    acquired = False
    if CLOUD_DEPLOYMENT:
        acquired = _cloud_job_lock.acquire(blocking=False)
        if not acquired:
            raise HTTPException(429, "Another video is processing. Open Previous jobs to check it, then try again.")
    job_id = uuid.uuid4().hex[:12]
    work = job_dir(job_id)
    try:
        if CLOUD_DEPLOYMENT:
            _cloud_purge_old_jobs(max_keep=1)
        work.mkdir(parents=True)
        suffix = Path(video.filename or "video.mp4").suffix.lower()
        if suffix not in (".mp4", ".mov", ".m4v", ".avi", ".webm"):
            raise HTTPException(400, "Video must be MP4, MOV, M4V, AVI or WebM")
        video_path = work / f"input{suffix}"
        limit = int(os.getenv("MAX_UPLOAD_MB", "600")) * 1024 * 1024
        written = 0
        with open(video_path, "wb") as f:
            while chunk := await video.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, f"Video exceeds {limit // 1024 // 1024} MB limit")
                f.write(chunk)
        if written == 0:
            raise HTTPException(400, "Empty video file")
        set_status(job_id, "queued", "Upload received")
        thread = threading.Thread(
            target=_run_cloud_job if CLOUD_DEPLOYMENT else process_job,
            args=(job_id, video_path, mode, quality, remove_noise == "true"),
            daemon=True,
        )
        thread.start()
        return {"id": job_id}
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        if acquired:
            _cloud_job_lock.release()
        raise


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return get_status(job_id)


@app.get("/api/jobs/{job_id}/model.ply")
def job_model(job_id: str, geometry: str = "original"):
    if geometry not in ("original", "surface"):
        raise HTTPException(400, "Unknown geometry type")
    path = job_dir(job_id) / ("surface.ply" if geometry == "surface" else "model.ply")
    if not path.is_file():
        raise HTTPException(404, "Requested geometry is not ready")
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


@app.get("/api/jobs/{job_id}/model.{format}")
def job_export(job_id: str, format: str, geometry: str = "original"):
    if CLOUD_DEPLOYMENT and format == "blend":
        raise HTTPException(422, "Blender export requires a server with Blender installed. Download OBJ/GLB and open it in Blender instead.")
    if format not in ("obj", "blend", "glb", "stl"):
        raise HTTPException(404, "Unsupported format")
    if geometry not in ("original", "surface"):
        raise HTTPException(400, "Unknown geometry type")
    if format in ("glb", "stl") and geometry != "surface":
        raise HTTPException(422, f".{format} export needs a reconstructed triangle surface")
    work = job_dir(job_id)
    source = work / ("surface.ply" if geometry == "surface" else "model.ply")
    if not source.is_file():
        raise HTTPException(404, "Requested geometry is not ready")
    prefix = "surface" if geometry == "surface" else "model"
    target = work / (prefix + "." + format)
    try:
        if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
            if format == "obj":
                export_obj(source, target)
            elif format == "blend":
                obj = work / (prefix + ".obj")
                if not obj.exists() or obj.stat().st_mtime < source.stat().st_mtime:
                    export_obj(source, obj)
                export_blend(obj, target)
            else:
                export_mesh_format(source, target, format)
    except Exception as exc:
        raise HTTPException(503, str(exc)) from exc
    return FileResponse(target, media_type="application/octet-stream", filename=prefix + "." + format)


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str):
    inputs = list(job_dir(job_id).glob("input.*"))
    if not inputs:
        raise HTTPException(404, "Source video not found")
    from mimetypes import guess_type
    content_type = guess_type(inputs[0].name)[0] or "application/octet-stream"
    return FileResponse(inputs[0], media_type=content_type)


@app.get("/api/jobs/{job_id}/quality")
def job_quality(job_id: str):
    path = job_dir(job_id) / "quality.json"
    if not path.is_file():
        raise HTTPException(404, "Quality report not ready")
    return json.loads(path.read_text())


@app.get("/api/jobs")
def recent_jobs(limit: int = 15):
    """Local machine job history (not a multi-user cloud service)."""
    limit = max(1, min(50, limit))
    rows = []
    for folder in sorted(JOBS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not folder.is_dir() or not (folder / "status.json").is_file():
            continue
        try:
            data = json.loads((folder / "status.json").read_text())
            rows.append({k: data.get(k) for k in ("id", "stage", "message", "updated", "model_type", "has_surface", "has_model")})
        except (ValueError, OSError):
            continue
        if len(rows) >= limit:
            break
    return {"jobs": rows}


@app.get("/api/jobs/{job_id}/frames")
def job_frames(job_id: str):
    folder = job_dir(job_id) / "frames"
    if not folder.exists():
        raise HTTPException(404, "No frames")
    return {"frames": sorted(p.name for p in folder.glob("*.jpg"))}


@app.get("/api/jobs/{job_id}/frames.zip")
def job_frames_zip(job_id: str):
    folder = job_dir(job_id) / "frames"
    if not folder.exists():
        raise HTTPException(404, "No frames")
    archive = shutil.make_archive(str(job_dir(job_id) / "frames_selected"), "zip", folder)
    return FileResponse(archive, media_type="application/zip", filename="frames.zip")


@app.get("/api/jobs/{job_id}/frames/{name}")
def job_frame(job_id: str, name: str):
    path = job_dir(job_id) / "frames" / Path(name).name
    if not path.exists():
        raise HTTPException(404, "Frame not found")
    return FileResponse(path, media_type="image/jpeg")


# Serve mobile UI and API on the same HTTPS origin: no Node or /api proxy required in production.
STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend" / "public"
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
