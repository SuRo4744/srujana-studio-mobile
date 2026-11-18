"""Run `python -m pytest -q` inside backend/ after installing requirements."""
from pathlib import Path

import cv2
import numpy as np
import trimesh
from fastapi.testclient import TestClient

import main
from exporters import export_obj
from quality import analyze_video
from surface import clean_occupancy, export_surface


def test_surface_mesh_and_exports(tmp_path):
    z, x, y = np.indices((28, 28, 28))
    mask = ((z - 14) ** 2 + (x - 14) ** 2 + (y - 14) ** 2 < 11 ** 2)
    mask[1, 1, 1] = True
    clean = clean_occupancy(mask)
    assert clean.sum() == mask.sum() - 1
    points = np.array([[0, 0, 0], [0.4, 0.2, -0.2]], dtype=np.float32)
    rgb = np.array([[230, 30, 20], [40, 170, 200]], dtype=np.uint8)
    surface = tmp_path / "surface.ply"
    result = export_surface(clean, points, rgb, 14, 1, surface)
    mesh = trimesh.load(surface, process=False)
    assert result["faces"] > 1000
    assert mesh.is_watertight
    assert mesh.visual.vertex_colors.shape[0] == result["vertices"]
    obj = tmp_path / "surface.obj"
    export_obj(surface, obj)
    assert "\nf " in obj.read_text()


def test_video_analysis(tmp_path):
    video = tmp_path / "test.avi"
    out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 12, (160, 120))
    assert out.isOpened()
    for i in range(24):
        image = np.full((120, 160, 3), 90, dtype=np.uint8)
        cv2.rectangle(image, (30 + i // 3, 30), (100 + i // 3, 90), (240, 140, 20), -1)
        out.write(image)
    out.release()
    result = analyze_video(video)
    assert result["width"] == 160
    assert result["frames_sampled"] > 0
    assert result["warnings"]


def test_api_surface_geometry_and_history(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "JOBS_DIR", tmp_path)
    folder = tmp_path / "test123"
    folder.mkdir()
    mesh = trimesh.creation.icosphere(subdivisions=1)
    mesh.visual.vertex_colors = [180, 60, 35, 255]
    mesh.export(folder / "surface.ply")
    mesh.export(folder / "model.ply")
    (folder / "quality.json").write_text('{"warnings":[]}')
    main.set_status("test123", "done", "Ready", has_model=True, has_surface=True)
    client = TestClient(main.app)
    response = client.get("/api/jobs/test123/model.ply?geometry=surface")
    assert response.status_code == 200
    assert response.content.startswith(b"ply")
    for ext in ("obj", "glb", "stl"):
        response = client.get(f"/api/jobs/test123/model.{ext}?geometry=surface")
        assert response.status_code == 200, response.text
        assert response.content
    assert client.get("/api/jobs/test123/model.stl").status_code == 422
    assert client.get("/api/jobs/test123/quality").json() == {"warnings": []}
    assert client.get("/api/jobs").json()["jobs"][0]["id"] == "test123"


def test_upload_route_returns_job_id(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "JOBS_DIR", tmp_path)
    # Prevent slow reconstruction in a route smoke test.
    monkeypatch.setattr(main, "process_job", lambda *args: None)
    client = TestClient(main.app)
    response = client.post(
        "/api/jobs", files={"video": ("capture.mp4", b"test-bytes", "video/mp4")},
        data={"mode": "turntable", "quality": "fast", "remove_noise": "true"},
    )
    assert response.status_code == 200, response.text
    job_id = response.json()["id"]
    assert len(job_id) == 12
    assert client.get(f"/api/jobs/{job_id}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}/video").content == b"test-bytes"
    assert client.post(
        "/api/jobs", files={"video": ("capture.mp4", b"abc", "video/mp4")},
        data={"mode": "turntable", "quality": "invalid"}
    ).status_code == 400
