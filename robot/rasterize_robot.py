import os
import torch
import torch.nn.functional as F
import collections
import math
import cv2
import numpy as np
import nvdiffrast.torch as dr
import trimesh
from tqdm import tqdm
from colmap_readers import readColmapSceneInfo
from PIL import Image

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

class COLMAPImage(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def write_cameras_text(cameras, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = "# Camera list with one line of data per camera:\n" + \
             "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" + \
             "# Number of cameras: {}\n".format(len(cameras))
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")

def write_images_text(images, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    with open(path, "w") as fid:
        # fid.write(HEADER)
        for _, img in images.items():
            image_header = [img.id, *img.qvec, *img.tvec, img.camera_id, img.name]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = []
            fid.write(" ".join(points_strings) + "\n")

def write_points3D_text(points3D, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    with open(path, "w") as fid:
        pass

def write_model(cameras, images, points3D, path, ext=".txt"):
    if ext == ".txt":
        write_cameras_text(cameras, os.path.join(path, "cameras" + ext))
        write_images_text(images, os.path.join(path, "images" + ext))
        write_points3D_text(points3D, os.path.join(path, "points3D") + ext)
    else:
        raise NotImplementedError
    return cameras, images, points3D

def build_colmap_cameras(f, cx, cy, height, width):
    params = [f, f, cx, cy]
    camera_id = 1
    cameras = {}
    cameras[camera_id] = Camera(id=camera_id,
                                model="PINHOLE",
                                width=int(width),
                                height=int(height),
                                params=np.array(params))
    return cameras

def build_colmap_images(view_matrices):
    '''
    :param view_matrices: [N, 4, 4], dtype: np.ndarray
    '''
    images = {}
    camera_id = 1
    for vid in range(view_matrices.shape[0]):
        image_id = vid + 1
        qvec = rotmat2qvec(view_matrices[vid, :3, :3])
        tvec = view_matrices[vid, :3, 3]
        images[image_id] = COLMAPImage(
            id=image_id, qvec=qvec, tvec=tvec,
            camera_id=camera_id, name="{:04}.png".format(vid),
            xys=None, point3D_ids=None)
    return images

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
        
        mv = cameraPs[v]
        campos = np.linalg.inv(cameraPs[v])[:3, 3]
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
    root_folder = "/data/guangyu/aRobotics/data/robot/mmk2/avg_link_1"
    vis_normal_path = os.path.join(root_folder, 'buffers', 'vis_n')
    res_normal_path = os.path.join(root_folder, 'buffers', 'normals')
    res_depth_path = os.path.join(root_folder, 'buffers', 'depths')
    res_image_path = os.path.join(root_folder, 'buffers', 'images')
    res_pose_path = os.path.join(root_folder, 'buffers/sparse')
    os.makedirs(vis_normal_path, exist_ok=True)
    os.makedirs(res_normal_path, exist_ok=True)
    os.makedirs(res_depth_path, exist_ok=True)
    os.makedirs(res_image_path, exist_ok=True)
    os.makedirs(res_pose_path, exist_ok=True)
    img_path = os.path.join(root_folder, "images")
    cam_path = root_folder
    mesh_path = os.path.join(root_folder, "mesh", 'new_avg_link.obj')
    
    cam_infos = readColmapSceneInfo(path=cam_path)
    # cam_infos = cam_infos[:10]
    
    mesh = trimesh.load_mesh(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()
    print('Successfully loading mesh')
    mesh.export(os.path.join(root_folder, 'buffers/1.obj'))
    verts = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces).int().cuda()
    normals = torch.from_numpy(mesh.vertex_normals.copy()).float().cuda()
    img_items = sorted(os.listdir(img_path))

    colmap_view_mats = []
    glctx = dr.RasterizeGLContext()
    for i, cinfo in tqdm(enumerate(cam_infos)):
        mvs, mvps, _ = mvsnet_to_dr(cinfo.intr[None, ...], cinfo.extr[None], sizes=[(cinfo.height, cinfo.width)], ss_ratio=1.0, zn=0.001, zf=1000.0)
        # print('\n')
        # print(mvs)
        # print(mvps)
        # breakpoint()
        colmap_view_mats.append(mvs)
        mvs = torch.from_numpy(mvs).float().cuda()
        mvps = torch.from_numpy(mvps).float().cuda()
        v_pos_cam = torch.matmul(torch.nn.functional.pad(verts, pad=(0,1), mode='constant', value=1.0), torch.transpose(mvs, 1, 2))
        v_pos_clip = torch.matmul(torch.nn.functional.pad(verts, pad=(0,1), mode='constant', value=1.0), torch.transpose(mvps, 1, 2))
        rast, rast_db = dr.rasterize(glctx, v_pos_clip, faces, (cinfo.height, cinfo.width))  # [N_v, H, W, 4]
        inlier_mask = rast[..., 3] > 0
        alpha = (inlier_mask[0].cpu().numpy() * 255.).astype('uint8')
        frg_normal, _ = dr.interpolate(normals, rast, faces)        # [N_v, H, W, 3]
        frg_normal = torch.nn.functional.normalize(frg_normal, p=2, dim=-1, eps=1e-8).contiguous()
        
        # world_view_transform = torch.tensor(getWorld2View2(cinfo.extr[:3, :3], cinfo.extr[:3, 3])).transpose(0, 1).cuda()
        # frg_normal = frg_normal @ world_view_transform[:3, :3]

        frg_depth, _ = dr.interpolate(v_pos_cam, rast, faces)           # shape: (N_v, H, W, 4)
        sav_depth = -frg_depth[0, ..., 2:3].cpu()

        torch.save(sav_depth, os.path.join(res_depth_path, "{}.pt".format(cinfo.image_name.split('.')[0])))
        torch.save(frg_normal[0].cpu(), os.path.join(res_normal_path, "{}.pt".format(cinfo.image_name.split('.')[0])))
        
        img = cv2.imread(os.path.join(img_path, img_items[i]))
        vis_n = (frg_normal[0].cpu().numpy() * 0.5 + 0.5) * 255.
        vis = img * 0.7 + vis_n * 0.3
        cv2.imwrite(os.path.join(vis_normal_path, img_items[i]), vis.astype('uint8'))
        
        img = torch.from_numpy(img).float()
        img[~inlier_mask[0].cpu()] = 0.0
        cv2.imwrite(os.path.join(res_image_path, img_items[i]), img.numpy().astype('uint8'))
        
        # vis_depth = frg_depth[0, ..., 2:3].cpu().numpy()
        # vis_depth -= vis_depth[inlier_mask[0].cpu().numpy()].min()
        # vis_depth /= vis_depth[inlier_mask[0].cpu().numpy()].max()
        # vis_depth = (vis_depth * 255.).astype('uint8')
        # cv2.imwrite(os.path.join(res_depth_path, img_items[i]), vis_depth)

    colmap_view_mats = np.concatenate(colmap_view_mats, axis=0)
    colmap_view_mats[:, 1:3] *= -1
    colmap_cameras = build_colmap_cameras(f=cam_infos[0].intr[0, 0], cx=cam_infos[0].intr[0, 2], cy=cam_infos[0].intr[1, 2], height=cam_infos[0].height, width=cam_infos[0].width)
    colmap_images = build_colmap_images(view_matrices=colmap_view_mats)
    write_model(colmap_cameras, colmap_images, points3D=None, path=res_pose_path)