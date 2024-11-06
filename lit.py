import os
import bpy
import datetime
import math
import random
import numpy as np
from typing import Tuple
from mathutils import Vector, Matrix, Quaternion

def get_3x4_RT_matrix_from_blender(cam: bpy.types.Object) -> Matrix:
    """Returns the 3x4 RT matrix from the given camera.

    Taken from Zero123, which in turn was taken from
    https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py

    Args:
        cam (bpy.types.Object): The camera object.

    Returns:
        Matrix: The 3x4 RT matrix from the given camera.
    """
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]

    R_bcam2cv = Matrix(
        ((1, 0, 0),
         (0, -1, 0),
         (0, 0, -1))
    )

    R_world2bcam = rotation.to_matrix().transposed()

    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    R_world2bcam = R_bcam2cv @ R_world2bcam
    T_world2bcam = R_bcam2cv @ T_world2bcam

    # put into 3x4 matrix
    RT = Matrix(
        (
            R_world2bcam[0][:] + (T_world2bcam[0],),
            R_world2bcam[1][:] + (T_world2bcam[1],),
            R_world2bcam[2][:] + (T_world2bcam[2],),
        )
    )

    # rotate around x-axis by 90-degree
    rotx90 = Matrix(
        (
            (1, 0, 0, 0),
            (0, 0, -1, 0),
            (0, 1, 0, 0),
            (0, 0, 0, 1)
        )
    )
    RT = RT @ rotx90

    # # colmap to open-gl
    # matcv2gl = Matrix(
    #     (
    #         (1, 0, 0),
    #         (0, -1, 0),
    #         (0, 0, -1)
    #     )
    # )
    # RT = matcv2gl @ RT
    return RT

def sample_random_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )

def sample_uniform_point_on_sphere(radius: float, num_horiz: int, num_verti: int):
    rets = []
    for j in range(num_verti):
        phi = ((j+1) / num_verti) * math.pi
        for i in range(num_horiz):
            theta = ((i+1) / num_horiz) * math.pi * 2
            rets.append((
                radius * math.sin(phi) * math.cos(theta),
                radius * math.sin(phi) * math.sin(theta),
                radius * math.cos(phi),
            ))
    return rets

def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")

def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

def setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    cam.data.sensor_height = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint

def add_lighting(lit_strength=1.0) -> None:
    # delete the default light
    if "Light" in bpy.data.objects.keys():
        bpy.data.objects["Light"].select_set(True)
        bpy.ops.object.delete()

    location_x = 0

    lit_path = "/media/womoer/Wdata/aRobotics/Relit/1.exr"
    lit_obj = bpy.data.images.load(lit_path)

    environment_texture_node = world_node_tree.nodes.new(type="ShaderNodeTexEnvironment")
    environment_texture_node.image = lit_obj
    location_x += 300

    background_node = world_node_tree.nodes.new(type="ShaderNodeBackground")
    background_node.inputs["Strength"].default_value = lit_strength
    background_node.location.x = location_x
    location_x += 300

    world_output_node = world_node_tree.nodes.new(type="ShaderNodeOutputWorld")
    world_output_node.location.x = location_x

    from_node = environment_texture_node
    to_node = background_node
    world_node_tree.links.new(from_node.outputs["Color"], to_node.inputs["Color"])

    from_node = background_node
    to_node = world_output_node
    world_node_tree.links.new(from_node.outputs["Background"], to_node.inputs["Surface"])

def render_image(object_path, out_path, num_horiz: int = 10, num_verti: int = 5, camera_dist: float = 1.5, lit_strength: float = 5.0) -> None:
    res_rgb_path = os.path.join(out_path, "images")
    res_pos_path = os.path.join(out_path, "poses")
    res_depth_path = os.path.join(out_path, "depths")
    os.makedirs(res_rgb_path, exist_ok=True)
    os.makedirs(res_pos_path, exist_ok=True)
    os.makedirs(res_depth_path, exist_ok=True)

    reset_scene()
    load_object(object_path)
    normalize_scene()

    # Create input render layer node
    render_layers = bpy.context.scene.node_tree.nodes.new('CompositorNodeRLayers')
    # Create depth output nodes
    depth_file_output = bpy.context.scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.base_path = ''
    depth_file_output.file_slots[0].use_node_format = True
    depth_file_output.format.file_format = "OPEN_EXR"
    bpy.context.scene.node_tree.links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])

    add_lighting(lit_strength=lit_strength)
    cam, cam_constraint = setup_camera()
    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty
    uniform_cam_points = sample_uniform_point_on_sphere(radius=camera_dist, num_horiz=num_horiz, num_verti=num_verti)
    for i in range(num_horiz * num_verti):
        # set the camera position
        cam.location = uniform_cam_points[i]

        direction = -cam.location
        rot_quat = direction.to_track_quat("-Z", "Y")
        cam.rotation_euler = rot_quat.to_euler()

        # render the image
        render_path = os.path.join(res_rgb_path, f"{i:03d}")
        scene.render.filepath = render_path
        depth_file_output.file_slots[0].path = os.path.join(res_depth_path, f"{i:03d}")
        bpy.ops.render.render(write_still=True)

        # save camera RT matrix
        rt_matrix = get_3x4_RT_matrix_from_blender(cam)
        rt_matrix_path = os.path.join(res_pos_path, f"{i:03d}.npy")
        np.save(rt_matrix_path, rt_matrix)

    bpy.ops.export_scene.obj(filepath=os.path.join(out_path, "1.obj"))

if __name__ == "__main__":
    context = bpy.context
    scene = context.scene
    render = scene.render

    render.engine = "CYCLES"
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = 512
    render.resolution_y = 512
    render.resolution_percentage = 100

    scene.use_nodes = True
    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True
    view_layer.use_pass_normal = True
    view_layer.use_pass_diffuse_color = True
    view_layer.use_pass_object_index = True
    scene.cycles.device = "GPU"
    scene.cycles.samples = 32
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.filter_width = 0.01
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True

    world_node_tree = bpy.context.scene.world.node_tree
    world_node_tree.nodes.clear()
    object_path = "/media/womoer/Wdata/aRobotics/robot_model/mmk2/avg_link/avg_link.obj"
    render_image(
        object_path=object_path,
        out_path="/media/womoer/Wdata/aRobotics/Relit/blender_output",
        num_horiz=10,
        num_verti=3,
        camera_dist=1.5,
        lit_strength=5.0,
    )