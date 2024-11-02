import os
import numpy as np
import cv2
import json
import math
from typing import NamedTuple
from colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text

class CameraInfo(NamedTuple):
    uid: int
    # R: np.array
    # T: np.array
    # FovY: np.array
    # FovX: np.array
    # image: np.array
    # image_path: str
    extr: np.array
    intr: np.array
    image_name: str
    width: int
    height: int
    
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def readColmapCameras(cam_extrinsics, cam_intrinsics):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        
        height = intr.height
        width = intr.width

        uid = intr.id
        # R = np.transpose(qvec2rotmat(extr.qvec))
        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec)
        
        extrinsic = np.concatenate((R, T[:, None]), axis=-1)

        intrinsic = np.array([
            [intr.params[0], 0.0, intr.params[2]],
            [0.0, intr.params[1], intr.params[3]],
            [0.0, 0.0, 1.0]
        ])

        # if intr.model=="SIMPLE_PINHOLE":
        #     focal_length_x = intr.params[0]
        #     FovY = focal2fov(focal_length_x, height)
        #     FovX = focal2fov(focal_length_x, width)
        # elif intr.model=="PINHOLE":
        #     focal_length_x = intr.params[0]
        #     focal_length_y = intr.params[1]
        #     FovY = focal2fov(focal_length_y, height)
        #     FovX = focal2fov(focal_length_x, width)
        # else:
        #     assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        # image_path = os.path.join(images_folder, os.path.basename(extr.name))
        # image_name = os.path.basename(image_path).split(".")[0]
        # image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, extr=extrinsic, intr=intrinsic, image_name=os.path.basename(extr.name), width=width, height=height)
        cam_infos.append(cam_info)
    return cam_infos

def readColmapSceneInfo(path, test_ids=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images"
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    return cam_infos