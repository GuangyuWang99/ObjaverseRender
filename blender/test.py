import os
import collections
import struct
import torch
import math
import cv2
import numpy as np
import nvdiffrast.torch as dr
import trimesh
from tqdm import tqdm
from PIL import Image

flip_mat = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

def cv_to_gl(cv):
    gl = cv @ flip_mat  # convert to GL convention used in iNGP
    return gl

if __name__ == "__main__":
    intrinsics = np.array(
        [[100.0, 0.0, 256.0],
         [0.0, 100.0, 256.0],
         [0.0, 0.0, 1.0]], dtype=np.float32)
    
    # colmap_mat = np.array(
    #     [[-5.8779e-01,  8.0902e-01,  0.0000e+00,  0.0000e+00],
    #     [ 5.7206e-01,  4.1563e-01, -7.0711e-01,  1.1921e-07],
    #     [-5.7206e-01, -4.1563e-01, -7.0711e-01,  2.5000e+00]], dtype=np.float32)
    
    colmap_mat = np.array(
        [[-9.5106e-01, -3.0902e-01,  0.0000e+00,  0.0000e+00],
        [-2.1851e-01,  6.7250e-01, -7.0711e-01,  2.3842e-07],
        [ 2.1851e-01, -6.7250e-01, -7.0711e-01,  2.5000e+00]], dtype=np.float32)
    
    colmap_proj = np.matmul(intrinsics, colmap_mat)
    out = cv2.decomposeProjectionMatrix(colmap_proj)
    R = out[1]
    t = out[2]
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R.transpose()
    c2w[:3, 3] = (t[:3] / t[3])[:, 0]
    
    c2w_gl = cv_to_gl(c2w)
    mv = np.linalg.inv(c2w_gl)
    print(mv)