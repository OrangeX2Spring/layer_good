"""Blender 4.x: --background --python tools/kvt_blender.py -- --artifacts DIR --mode reveal --out FILE.blend."""
import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--mode", choices=["actual", "reveal"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--point-radius-m", type=float, required=True)
    parser.add_argument("--display-stride", type=int, default=1, help="Display-only subsampling; measurements remain full density")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    assert args.point_radius_m > 0 and args.display_stride > 0
    if args.out.exists():
        raise FileExistsError(args.out)
    playback = json.loads((args.artifacts / "playback.json").read_text())
    entries = playback[args.mode]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, len(entries)
    scene["playback_kind"] = args.mode
    scene["note"] = (playback["reveal_note"] if args.mode == "reveal" else
                     "Actual reconstruction snapshots, independently aligned to fixed reference anchors")
    scene["display_stride"] = args.display_stride
    scene["point_radius_m"] = args.point_radius_m
    material = bpy.data.materials.new("Point RGB")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    attribute = nodes.new("ShaderNodeAttribute")
    attribute.attribute_name = "point_rgb"
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(attribute.outputs["Color"], emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs[0], output.inputs["Surface"])
    # Small octahedra are visible in both the viewport and render engines without add-ons.
    offsets = np.concatenate((np.eye(3), -np.eye(3))) * args.point_radius_m
    triangles = np.array([[0, 1, 2], [0, 2, 4], [0, 4, 5], [0, 5, 1],
                          [3, 2, 1], [3, 4, 2], [3, 5, 4], [3, 1, 5]])
    expected_vertices = []
    for step, entry in enumerate(entries, 1):
        with np.load(args.artifacts / entry["file"]) as cloud:
            xyz = cloud["xyz"][::args.display_stride]
            rgb = cloud["rgb"][::args.display_stride] / 255.0
        vertices = (xyz[:, None, :] + offsets).reshape(-1, 3)
        faces = (np.arange(len(xyz))[:, None, None] * 6 + triangles).reshape(-1, 3)
        mesh = bpy.data.meshes.new(f"points_{step:04d}")
        mesh.from_pydata(vertices.tolist(), [], faces.tolist())
        mesh.update()
        colors = mesh.color_attributes.new(name="point_rgb", type="FLOAT_COLOR", domain="POINT")
        # Convert input sRGB bytes to linear values for Blender's shader attributes.
        rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        rgba = np.column_stack((np.repeat(rgb, 6, axis=0), np.ones(len(vertices))))
        colors.data.foreach_set("color", rgba.ravel())
        mesh.materials.append(material)
        obj = bpy.data.objects.new(f"{args.mode}_{step:04d}_source_{entry['source_frame']}", mesh)
        scene.collection.objects.link(obj)
        obj["source_frame"] = entry["source_frame"]
        obj["display_points"] = len(xyz)
        expected_vertices.append(len(vertices))
        transitions = [(0, True), (step, False)]
        if args.mode == "actual":
            transitions.append((step + 1, True))
        for frame, hidden in transitions:
            obj.hide_viewport = hidden
            obj.hide_render = hidden
            obj.keyframe_insert(data_path="hide_viewport", frame=frame)
            obj.keyframe_insert(data_path="hide_render", frame=frame)
    scene.frame_set(len(entries))
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.shading.type = "MATERIAL"
                area.spaces.active.clip_end = 1000
    args.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out.resolve()))
    bpy.ops.wm.open_mainfile(filepath=str(args.out.resolve()))
    objects = sorted(bpy.context.scene.objects, key=lambda obj: obj.name)
    assert [len(obj.data.vertices) for obj in objects] == expected_vertices
    for frame in range(1, len(entries) + 1):
        bpy.context.scene.frame_set(frame)
        visible = sum(not obj.hide_viewport for obj in objects)
        assert visible == (frame if args.mode == "reveal" else 1)
    print(f"BLENDER OK: reloaded {len(entries)} clouds and verified timeline visibility")


if __name__ == "__main__":
    main()
