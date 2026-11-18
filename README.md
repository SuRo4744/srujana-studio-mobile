# Srujana Studio Pro — Video to 3D (frontend + backend)

This edition upgrades the existing Srujana Studio project; it **does not** use Spirula Studio.

## What's new

| Feature | Available for | Notes |
|---|---|---|
| Turntable quality: fast / balanced / high | Turntable | 96 / 160 / 192 voxel grid; high quality uses more memory and time |
| Voxel noise cleanup | Turntable | Largest 26-connected body, optional toggle |
| Colored triangle-mesh reconstruction | Turntable | Marching cubes of carved occupancy, smoothing, color transfer from points |
| Points / solid / wireframe viewer | Turntable | Real triangles in solid and wireframe; no invented geometry for sparse clouds |
| Adaptive point clarity | Point / voxel models | Adjustable point size, optional cap to 3.5 on-screen pixels; round point sprites |
| Viewer customization | Point / voxel / triangle-mesh models | Brightness, opacity, surface finish, X/Y/Z rotation and scale, background, zoom, screenshot |
| Save customizer preset | Browser | Saves **preview settings** in your browser's localStorage |
| Video diagnostics | All modes | Samples frames for resolution, duration, exposure and indicative sharpness |
| Reopen recent jobs | This machine | Uses job folders saved on the backend; source video may be compared side-by-side |
| Original downloads | Any model | `.ply`, vertex-based `.obj`, `.blend` (Blender installation required) |
| Surface downloads | Turntable with mesh | `.ply`, `.obj`, `.blend`, `.glb`, `.stl` based on real triangles |

**Quality limitations:** Turntable mode is silhouette carving, not photogrammetric texture recovery. Its solid surface is an *approximate visual hull*, not necessarily a precise or print-ready scan. Thin handles, concavities, reflective material, and parts hidden in the input video may be incorrect. Surface smoothing does not recover missing texture detail. STL discards color; OBJ vertex-color support varies by app. Blender exports require Blender installed on the backend.

**Orbit mode:** `auto` uses Gaussian splatting only when Nerfstudio/COLMAP/CUDA tools are installed. Otherwise `pycolmap` can produce a sparse CPU point cloud, if separately installed. We **do not** triangulate sparse point clouds or splats into speculative surfaces; GLB/STL appear only when a proper triangle mesh is available. The advanced point/material customizer does not apply to the separate Gaussian splat renderer.

## Requirements

- macOS (including MacBook Air) or Windows/Linux
- Python 3.10–3.13 with `venv` / pip
- Node.js 18+ and npm
- For `.blend`: [Blender](https://www.blender.org/download/) installed, normally `/Applications/Blender.app` on macOS. If installed elsewhere, add its executable to your `PATH`.
- Internet access in browser for Three.js and the Gaussian-splat viewer modules loaded from jsDelivr.

## First-time setup on macOS (two terminals)

From **project root**, open Terminal A:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

Open Terminal B:

```bash
cd frontend
npm ci
npm start
```

Open **http://localhost:3000** (FastAPI runs on `http://localhost:8000`).

### After the first install

Terminal A:

```bash
cd backend
source .venv/bin/activate
python main.py
```

Terminal B:

```bash
cd frontend
npm start
```

## How to use

1. Capture a steady video. For **Turntable**, keep the camera still, rotate the object a complete turn, with a plain, uniformly lit background. In **Orbit**, keep the object still while moving around it.
2. Select the video, capture style and turntable reconstruction quality. Keep *Remove disconnected voxel noise* checked unless parts detach incorrectly.
3. Click **Create 3D Model**. The backend checks video quality, selects frames/reconstructs, writes the original `model.ply`, and when possible creates `surface.ply`.
4. On turntable models, switch **Display** among Sharp points, Solid surface and Wireframe. Solid is selected by default. Use the customizer and optional video overlay to check the reconstruction.
5. Click **Download 3D Model ▾**. The original PLY is unchanged. For OBJ/Blender choose Original vertices or Triangle surface. GLB/STL only appear when triangulated surface is available.
6. Use **Previous jobs** to revisit results after restarting the browser (as long as the backend's `jobs/` folder remains).

## Structure

```text
srujana-studio-pro/
  frontend/
    server.js                 # Express site + corrected /api proxy
    package.json, package-lock.json
    public/
      index.html
      styles.css
      app.js                  # 3D previews, customizer, export controls and history
  backend/
    main.py                   # Original video pipeline + extended API
    quality.py                # Video preflight samples
    surface.py                # Occupancy cleanup + colored marching cubes
    exporters.py              # OBJ, Blender, GLB, STL exporters
    requirements.txt
    test_features.py
    jobs/                     # Created at runtime; uploads, status, exports
  README.md
```

## Useful API routes

- `POST /api/jobs` multipart form with `video`, `mode`, `quality` and `remove_noise`
- `GET /api/jobs` list recent jobs (local environment; no login/security layer)
- `GET /api/jobs/{id}` status, including `has_surface`, `quality_report` and model statistics
- `GET /api/jobs/{id}/video` source video for comparison
- `GET /api/jobs/{id}/quality` video diagnostics
- `GET /api/jobs/{id}/model.ply` original model
- `GET /api/jobs/{id}/model.ply?geometry=surface` triangle mesh (when available)
- `GET /api/jobs/{id}/model.obj?geometry=surface` mesh OBJ
- `GET /api/jobs/{id}/model.blend?geometry=surface` Blender project (requires Blender executable)
- `GET /api/jobs/{id}/model.glb?geometry=surface` colored surface GLB
- `GET /api/jobs/{id}/model.stl?geometry=surface` colorless STL mesh

Use `?geometry=original` (default) for the original point-cloud OBJ or Blender export. If a model has no surface, the mesh formats return a clear 404/422 instead of inventing faces.

## Tests

```bash
cd frontend && npm run check
cd ../backend && python -m pip install pytest httpx
python -m pytest -q test_features.py
```

## Troubleshooting

- **`zsh: command not found: pip`**: use `python3 -m pip` (or activate `.venv` and use `python -m pip`).
- **`POST /jobs 404`**: the proxy in `frontend/server.js` must restore `/api` in `pathRewrite`. This project includes that fix. Restart both terminals after replacing the project.
- **`DEP0060 util._extend` warning**: this comes from a Node dependency; it does not prevent upload.
- **`No rotation detected`**: record a full, slow turntable spin with a static camera.
- **`Could not separate the object from the backdrop`**: improve the background contrast and lighting. The built-in silhouette segmentation is **not AI background removal**.
- **Blender export 503**: install Blender and check `/Applications/Blender.app/Contents/MacOS/Blender` exists or add Blender to `PATH`.
- **No solid surface option**: only turntable models with a valid occupancy volume can be triangulated in this edition. GPU splats and sparse orbit clouds retain their existing preview modes.
- **Missing JS module in browser**: connect to internet, as Three.js modules load from a CDN.

**Privacy:** The application runs locally by default and saves uploads in `backend/jobs/`. Its default backend binds to `0.0.0.0`, and there is no authentication. Do not expose ports 3000/8000 publicly without adding authentication, upload limits, and a proper deployment configuration.
