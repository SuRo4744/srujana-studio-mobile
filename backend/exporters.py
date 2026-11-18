"""Export the reconstruction's PLY vertices to OBJ or a native Blender project.

OBJ is a vertex-only point cloud when the reconstruction has no faces. It cannot
preserve Gaussian-splat scale/rotation/opacity; the original PLY is authoritative.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile


def export_obj(source: Path, target: Path) -> Path:
    import trimesh
    data = trimesh.load(str(source), process=False)
    if isinstance(data, trimesh.Scene):
        geometries = list(data.geometry.values())
        if not geometries:
            raise ValueError('PLY has no geometry')
        data = geometries[0]
    vertices = data.vertices
    if len(vertices) == 0:
        raise ValueError('PLY has no vertices')
    colors = getattr(data.visual, 'vertex_colors', None)
    with target.open('w', encoding='utf-8') as f:
        f.write('# Srujana Studio OBJ: point cloud vertices (not an invented mesh)\n')
        for i, (x, y, z) in enumerate(vertices):
            if colors is not None and i < len(colors):
                r, g, b = colors[i][:3]
                f.write(f'v {x:.8g} {y:.8g} {z:.8g} {r/255:.6g} {g/255:.6g} {b/255:.6g}\n')
            else:
                f.write(f'v {x:.8g} {y:.8g} {z:.8g}\n')
        faces = getattr(data, 'faces', None)
        if faces is not None and len(faces):
            for face in faces:
                f.write('f ' + ' '.join(str(int(v) + 1) for v in face) + '\n')
    return target


BLENDER_SCRIPT = r'''
import bpy
import sys
from pathlib import Path
args = sys.argv[sys.argv.index('--') + 1:]
obj_path, output_path = map(Path, args)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
vertices, faces, colors = [], [], []
for line in obj_path.open(encoding='utf-8'):
    parts = line.split()
    if not parts: continue
    if parts[0] == 'v':
        vertices.append(tuple(map(float, parts[1:4])))
        colors.append(tuple(map(float, parts[4:7])) if len(parts) >= 7 else (0.8, 0.8, 0.8))
    elif parts[0] == 'f':
        faces.append(tuple(int(p.split('/')[0])-1 for p in parts[1:]))
mesh = bpy.data.meshes.new('Reconstructed geometry')
mesh.from_pydata(vertices, [], faces)
mesh.update()
obj = bpy.data.objects.new('Srujana 3D Model', mesh)
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
if vertices:
    attr = mesh.color_attributes.new(name='Color', type='FLOAT_COLOR', domain='POINT')
    for i, c in enumerate(colors): attr.data[i].color = (*c, 1.0)
    mat = bpy.data.materials.new('Vertex colors')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    color_node = nodes.new('ShaderNodeVertexColor')
    color_node.layer_name = 'Color'
    bsdf = nodes.get('Principled BSDF')
    mat.node_tree.links.new(color_node.outputs['Color'], bsdf.inputs['Base Color'])
    mesh.materials.append(mat)
    if not faces:
        # Blender's mesh vertex-only objects are invisible in rendered mode.
        # Show the cloud in viewport as colored vertex dots (no fake faces).
        obj.display_type = 'WIRE'
        obj.show_in_front = True
bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
'''


def export_blend(obj_path: Path, target: Path) -> Path:
    blender = shutil.which('blender')
    if not blender and Path('/Applications/Blender.app/Contents/MacOS/Blender').is_file():
        blender = '/Applications/Blender.app/Contents/MacOS/Blender'
    if not blender:
        raise RuntimeError('Blender executable is not installed on the backend. Install Blender and add it to PATH to enable .blend export.')
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / 'make_blend.py'
        script.write_text(BLENDER_SCRIPT)
        proc = subprocess.run([blender, '--background', '--factory-startup', '--python', str(script), '--', str(obj_path), str(target)], capture_output=True, text=True, timeout=300)
        if proc.returncode or not target.exists():
            raise RuntimeError('Blender export failed: ' + (proc.stderr or proc.stdout)[-1500:])
    return target


def export_mesh_format(source: Path, target: Path, file_format: str) -> Path:
    """Export genuine triangles to GLB or STL; reject point-only geometry."""
    if file_format not in ("glb", "stl"):
        raise ValueError("Unsupported surface format")
    import trimesh
    mesh = trimesh.load(str(source), process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f".{file_format} needs real mesh faces. Turntable surface reconstruction is required.")
    mesh.export(str(target), file_type=file_format)
    return target
