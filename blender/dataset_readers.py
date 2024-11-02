#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import collections
from PIL import Image
from typing import NamedTuple
from colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
import numpy as np
import math
import json
from pathlib import Path

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    K: np.array
    FovY: np.array
    FovX: np.array
    # image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        # R = np.transpose(qvec2rotmat(extr.qvec))
        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec)

        assert intr.model=="PINHOLE"

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            K = np.array([
                [intr.params[0], 0., intr.params[2]],
                [0., intr.params[1], intr.params[3]],
                [0., 0., 1.]
            ])
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        # image = Image.open(image_path)

        cam_info = CameraInfo(
            uid=uid, R=R, T=T, K=K, FovY=FovY, FovX=FovX, 
            image_path=image_path, image_name=image_name, width=width, height=height
        )
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readMujocoCamInfo(path, images=None, ):
    MujocoCamera = collections.namedtuple(
        "MujocoCamera", ["id", "model", "width", "height", "params"])
    MujocoImage = collections.namedtuple(
        "MujocoImage", ["id", "qvec", "tvec", "camera_id", "name"])

    cameras_extrinsic_file = os.path.join(path, "images.txt")
    cameras_intrinsic_file = os.path.join(path, "cameras.txt")
    
    cam_extrinsics = {}
    with open(cameras_extrinsic_file, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                cam_extrinsics[image_id] = MujocoImage(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name)
    
    cam_intrinsics = {}
    with open(cameras_intrinsic_file, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cam_intrinsics[camera_id] = MujocoCamera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    return cam_infos