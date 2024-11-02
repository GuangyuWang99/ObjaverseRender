import os
import numpy as np
import cv2
import json
import math
import torch
import torch.nn.functional as F
from typing import NamedTuple
from colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, rotmat2qvec, \
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
        R = qvec2rotmat(extr.qvec[[1,2,3,0]])
        T = np.array(extr.tvec)
        
        rots = R.transpose()
        trans = -rots @ T[:, None]
        extrinsic = np.eye(4)
        extrinsic[:3] = np.concatenate((rots, trans), axis=-1)
        
        intrinsic = np.array([
            [intr.params[0], 0.0, intr.params[2]],
            [0.0, intr.params[1], intr.params[3]],
            [0.0, 0.0, 1.0]
        ])
        
        cam_info = CameraInfo(uid=uid, extr=extrinsic, intr=intrinsic, image_name=os.path.basename(extr.name), width=width, height=height)
        cam_infos.append(cam_info)
    return cam_infos

def readColmapSceneInfo(path, test_ids=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse", "camera.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse", "camera.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images"
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    return cam_infos