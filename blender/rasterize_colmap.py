import os
import torch
import math
import cv2
import numpy as np
import nvdiffrast.torch as dr
import trimesh
from tqdm import tqdm
from colmap_readers import readColmapSceneInfo

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
    mvs = []
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
        mvs.append(mv)
        mvps.append(mvp)
        cs.append(campos)
    mvs = np.stack(mvs, axis=0)
    mvps = np.stack(mvps, axis=0)
    cs = np.stack(cs, axis=0)
    return mvs, mvps, cs

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

if __name__ == "__main__":
    # root_folder = "/data/guangyu/aRobotics/data/sample"
    root_folder = "/data/guangyu/dataset/aLit/GreatWall/TW1_Ego"
    # res_normal_path = os.path.join(root_folder, 'normals_nvs')
    # res_depth_path = os.path.join(root_folder, 'depths_nvs')
    # vis_normal_path = os.path.join(root_folder, 'vis_n_nvs')
    vis_normal_path = os.path.join(root_folder, 'vis_n')
    res_normal_path = os.path.join(root_folder, 'normals')
    res_depth_path = os.path.join(root_folder, 'depths')
    os.makedirs(vis_normal_path, exist_ok=True)
    os.makedirs(res_normal_path, exist_ok=True)
    os.makedirs(res_depth_path, exist_ok=True)
    img_path = os.path.join(root_folder, "colmap/images")
    cam_path = os.path.join(root_folder, "colmap")
    # img_path = os.path.join(root_folder, "colmap_nvs/images")
    # cam_path = os.path.join(root_folder, "colmap_nvs")
    mesh_path = os.path.join(root_folder, '2.obj')
    
    cam_infos = readColmapSceneInfo(path=cam_path)
    
    mesh = trimesh.load_mesh(mesh_path)
    print('Successfully loading mesh')
    verts = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces).int().cuda()
    normals = torch.from_numpy(mesh.vertex_normals.copy()).float().cuda()
    img_items = sorted(os.listdir(img_path))

    glctx = dr.RasterizeGLContext()
    for i, cinfo in tqdm(enumerate(cam_infos)):
        mvs, mvps, _ = mvsnet_to_dr(cinfo.intr[None, ...], cinfo.extr[None], sizes=[(cinfo.height, cinfo.width)], ss_ratio=1.0, zn=0.001, zf=1000.0)
        mvs = torch.from_numpy(mvs).float().cuda()
        mvps = torch.from_numpy(mvps).float().cuda()
        v_pos_cam = torch.matmul(torch.nn.functional.pad(verts, pad=(0,1), mode='constant', value=1.0), torch.transpose(mvs, 1, 2))
        v_pos_clip = torch.matmul(torch.nn.functional.pad(verts, pad=(0,1), mode='constant', value=1.0), torch.transpose(mvps, 1, 2))
        rast, rast_db = dr.rasterize(glctx, v_pos_clip, faces, (cinfo.height, cinfo.width))  # [N_v, H, W, 4]
        inlier_mask = rast[..., 3] > 0
        alpha = (inlier_mask[0].cpu().numpy() * 255.).astype('uint8')
        frg_normal, _ = dr.interpolate(normals, rast, faces)        # [N_v, H, W, 3]
        frg_normal = torch.nn.functional.normalize(frg_normal, p=2, dim=-1, eps=1e-8).contiguous()
        
        world_view_transform = torch.tensor(getWorld2View2(cinfo.extr[:3, :3], cinfo.extr[:3, 3])).transpose(0, 1).cuda()
        frg_normal = frg_normal @ world_view_transform[:3, :3]

        frg_depth, _ = dr.interpolate(v_pos_cam, rast, faces)           # shape: (N_v, H, W, 4)
        sav_depth = -frg_depth[0, ..., 2:3].cpu()

        torch.save(sav_depth, os.path.join(res_depth_path, "{}.pt".format(cinfo.image_name.split('.')[0])))
        torch.save(frg_normal[0].cpu(), os.path.join(res_normal_path, "{}.pt".format(cinfo.image_name.split('.')[0])))
        
        img = cv2.imread(os.path.join(img_path, img_items[i]))
        vis_n = (frg_normal[0].cpu().numpy() * 0.5 + 0.5) * 255.
        vis = img * 0.7 + vis_n * 0.3
        cv2.imwrite(os.path.join(vis_normal_path, img_items[i]), vis.astype('uint8'))
        
        # vis_depth = frg_depth[0, ..., 2:3].cpu().numpy()
        # vis_depth -= vis_depth[inlier_mask[0].cpu().numpy()].min()
        # vis_depth /= vis_depth[inlier_mask[0].cpu().numpy()].max()
        # vis_depth = (vis_depth * 255.).astype('uint8')
        # cv2.imwrite(os.path.join(res_depth_path, img_items[i]), vis_depth)
        