# Deploy Srujana Studio Mobile (Render free)

This is a ready-to-deploy **single-service** FastAPI application. Frontend HTML, CSS, JS and backend Python remain in separate source folders, but the hosted Python process serves both via one origin. No Node process, proxy, or CORS setup is required on Render.

## Before you deploy

1. Create a **GitHub repository** and upload the contents of this folder **including `render.yaml`** to its root. Do not upload the ZIP *as one file*; upload the extracted contents.
2. Sign into [Render](https://dashboard.render.com/) and choose **New → Blueprint**, select that GitHub repository and click **Apply**.
3. When Render requests `STUDIO_PASSWORD`, create a strong private password. The username is **`studio`**. Never commit the password to GitHub.
4. Once the build is live, Render shows a public HTTPS URL such as `https://<your-service>.onrender.com`. That URL is determined **by Render**, not by the ZIP filename. Open it on your iPhone or Android device, then use username `studio` and your password. Optionally choose **Add to Home Screen** in your mobile browser.

If you connected Render and GitHub in ChatGPT, their connected actions may be able to perform steps 2–4 directly. Without a connected hosting account nobody can create a genuine public URL for your account.

## Free-tier limitations

- Only **Turntable** reconstruction with Fast and Balanced presets is available. 3D Gaussian GPU training, CPU COLMAP orbit reconstruction, and `.blend` export need a more capable host; on mobile, use the exported `.obj` or `.glb` with Blender on a PC.
- **25 MB** upload limit; film a short object rotation (8–20 seconds) on a plain background. Use H.264 MP4 for easiest decoding; Apple's HEVC MOV may fail depending on video codec.
- **One video at a time.** Free compute is limited to 512 MB of RAM / small CPU; reconstruction can be slow or may run out of memory for some videos.
- Free service sleeps after ~15 minutes without requests; opening the URL again wakes it (potential wait).
- Files on Render free service are **ephemeral**; previous jobs and uploaded videos can disappear after a restart/redeploy. Only two recent jobs are kept while service is running.
- Authentication is HTTP Basic (`studio` / `STUDIO_PASSWORD`) and applies to HTML, API requests, and downloaded files. `/health` is public for Render health checks. Use only the HTTPS Render URL.
- This is **not production-grade multi-user hosting** and it is not a guarantee that resource-heavy reconstruction can finish within free compute limits.

## Local smoke test (single server)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
CLOUD_DEPLOYMENT=1 STUDIO_PASSWORD=YOUR_TEST_PASSWORD MAX_UPLOAD_MB=25 TT_GRID=64 TT_VIEWS=16 TT_WIDTH=320 TT_MAX_FRAMES=80 python main.py
```
Open http://localhost:8000 and log in as `studio` / your chosen test password. This local HTTP address is for testing only.

## Diagnostics

- `/health` responds with `{ "status": "ok" }` (public).
- `/api/config` shows whether cloud features are active (password required).
- If the page says `STUDIO_PASSWORD` is missing, add that environment variable in the Render Dashboard and redeploy.
- If Render says **out of memory**, lower `CLOUD_MAX_GRID` and `TT_GRID` to `48`, `TT_WIDTH` to `256` and `TT_MAX_FRAMES` to `60` in `render.yaml` and redeploy; surface detail will decrease.
- If video processing fails with `Could not open the video file`, convert it to H.264 MP4 before upload.
- If you need full high-resolution 3D capture, persistent projects, and Blender exports, this requires a paid service or a computer running the backend.
