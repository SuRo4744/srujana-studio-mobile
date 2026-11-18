"""Turntable voxel cleanup and colored triangle-mesh extraction.

Only used when a real occupied-voxel volume exists. We deliberately do not
manufacture triangles from sparse SfM points or Gaussian-splat PLY files.
"""
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure
import trimesh


def clean_occupancy(occupancy: np.ndarray, min_component_ratio: float = 0.002) -> np.ndarray:
    """Remove isolated disconnected voxel components, retaining the largest body."""
    if occupancy.ndim != 3 or not occupancy.any():
        raise ValueError("A nonempty 3-D occupancy volume is required")
    labels, n = ndimage.label(occupancy, structure=np.ones((3, 3, 3), dtype=bool))
    if n <= 1:
        return occupancy.astype(bool, copy=True)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest = int(np.argmax(counts))
    # Retain the main object, drop disconnected floaters. Does not fill holes.
    return labels == largest


def export_surface(
    occupancy: np.ndarray,
    point_xyz: np.ndarray,
    point_rgb: np.ndarray,
    radius: float,
    voxel_size: float,
    target: Path,
    *,
    smooth_sigma: float = 0.75,
) -> dict:
    """Marching-cubes surface with colors transferred from the colored voxel PLY.

    occupied axis order = [vertical z, horizontal x, depth y]; the 3D viewer
    uses [x, z, -y] / radius. Vertex colors come from the reconstructed points;
    smoothing and interpolation do not restore missing footage detail.
    """
    if radius <= 0 or voxel_size <= 0:
        raise ValueError("Invalid voxel coordinate scale")
    if len(point_xyz) == 0 or len(point_rgb) != len(point_xyz):
        raise ValueError("Missing colored surface points")
    if not occupancy.any():
        raise ValueError("Empty voxel volume")
    # Zero-padding ensures even surfaces touching the edge are closed.
    padded = np.pad(occupancy.astype(np.float32), 2)
    if smooth_sigma > 0:
        padded = ndimage.gaussian_filter(padded, sigma=smooth_sigma)
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5)
    # 2 padded cells, plus original voxel-center offset 0.5
    vert_zi, vert_xi, vert_yi = (verts - 2.0).T
    x = (vert_xi + 0.5) * voxel_size - radius
    y = (vert_yi + 0.5) * voxel_size - radius
    z = (vert_zi + 0.5) * voxel_size
    xyz = np.column_stack([x, z, -y]).astype(np.float32) / radius
    # Marching cubes of an occupancy field has reliable topology but may have
    # reversed normals due to the axis permutation. trimesh.fix_normals handles it.
    tree = cKDTree(point_xyz)
    indices = tree.query(xyz, workers=1)[1]
    colors = np.asarray(point_rgb[indices], dtype=np.uint8)
    mesh = trimesh.Trimesh(vertices=xyz, faces=faces, process=False)
    mesh.visual.vertex_colors = np.column_stack([colors, np.full(len(colors), 255, dtype=np.uint8)])
    mesh.fix_normals()
    mesh.export(str(target), file_type="ply")
    return {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)), "watertight": bool(mesh.is_watertight)}
