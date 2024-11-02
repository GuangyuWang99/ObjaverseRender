import os
import torch
import math
import cv2
import numpy as np
import nvdiffrast.torch as dr
import trimesh
from tqdm import tqdm

flip_mat = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

def cv_to_gl(cv):
    gl = cv @ flip_mat  # convert to GL convention used in iNGP
    return gl

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def mvsnet_to_dr(cameraKs, cameraPs, sizes, ss_ratio, zn=0.001, zf=1000.0):
    mvps = []
    cs = []
    for v in range(cameraKs.shape[0]):
        fl_x = cameraKs[v][0][0]
        c_x  = cameraKs[v][0][2]
        fl_y = cameraKs[v][1][1]
        c_y  = cameraKs[v][1][2]
        H, W = sizes[v] if len(sizes) > 1 else sizes[0]
        fov_x = math.atan(W * ss_ratio / (fl_x * 2)) * 2
        fov_y = math.atan(H * ss_ratio / (fl_y * 2)) * 2
        x = np.tan(fov_x / 2)
        y = np.tan(fov_y / 2)
        # aspect = W / H
        proj = np.array([[1/x,       0,        (W - 2*c_x)/W,            0], 
                         [           0, 1/-y,  (H - 2*c_y)/H,            0], 
                         [           0,    0, -(zf+zn)/(zf-zn), -(2*zf*zn)/(zf-zn)], 
                         [           0,    0,           -1,              0]], dtype=np.float32)
        out = cv2.decomposeProjectionMatrix(cameraPs[v])
        R = out[1]
        t = out[2]
        # R = cameraRs[v]
        # t = cameraTs[v]
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R.transpose()
        c2w[:3, 3] = (t[:3] / t[3])[:, 0]
        # c2w[:3, 3] = t[:3]
        
        c2w_gl = cv_to_gl(c2w)
        mv = np.linalg.inv(c2w_gl)
        campos = c2w_gl[:3, 3]
        mvp = proj @ mv
        mvps.append(mvp)
        cs.append(campos)
    mvps = np.stack(mvps, axis=0)
    cs = np.stack(cs, axis=0)
    return mvps, cs

if __name__ == "__main__":
    root_folder = "/data/guangyu/aRobotics/data/sample"
    res_path = os.path.join(root_folder, 'normals')
    os.makedirs(res_path, exist_ok=True)
    img_path = os.path.join(root_folder, "images")
    pos_path = os.path.join(root_folder, "ego.npy")
    mesh_path = os.path.join(root_folder, '1.ply')
    
    fov = 61.9275 / 180 * np.pi
    H = 989
    W = 1320
    cx, cy = W / 2, H / 2
    focal = fov2focal(fov, W)
    print("focal: ", focal)
    intrinsics = np.array([
        [focal, 0, cx],
        [0, focal, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    extrinsics = np.load(pos_path)
    intrinsics = intrinsics[None, ...].repeat(extrinsics.shape[0], axis=0)
    w2cs = np.matmul(intrinsics, extrinsics)
    
    mesh = trimesh.load_mesh(mesh_path)
    print('Successfully loading mesh')
    verts = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces).int().cuda()
    normals = torch.from_numpy(mesh.vertex_normals.copy()).float().cuda()
    
    img_items = sorted(os.listdir(img_path))

    glctx = dr.RasterizeGLContext()
    for i in tqdm(range(w2cs.shape[0])):
        mvps, _ = mvsnet_to_dr(intrinsics[i: i+1], w2cs[i: i+1], sizes=[(H, W)], ss_ratio=1.0, zn=0.001, zf=1000.0)
        mvps = torch.from_numpy(mvps).float().cuda()
        v_pos_clip = torch.matmul(torch.nn.functional.pad(verts, pad=(0,1), mode='constant', value=1.0), torch.transpose(mvps, 1, 2))
        rast, rast_db = dr.rasterize(glctx, v_pos_clip, faces, (H, W))  # [N_v, H, W, 4]
        inlier_mask = rast[..., 3] > 0
        alpha = (inlier_mask[0].cpu().numpy() * 255.).astype('uint8')
        frg_normal, _ = dr.interpolate(normals, rast, faces)        # [N_v, H, W, 3]
        frg_normal = torch.nn.functional.normalize(frg_normal, p=2, dim=-1, eps=1e-8).contiguous()

        img = cv2.imread(os.path.join(img_path, img_items[i]))
        vis_n = (frg_normal[0].cpu().numpy() * 0.5 + 0.5) * 255.
        vis = img * 0.7 + vis_n * 0.3
        cv2.imwrite(os.path.join(res_path, img_items[i]), vis.astype('uint8'))
        