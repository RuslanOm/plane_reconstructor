import open3d as o3d
import numpy as np
import glob

from models.vit.util.io import read_pfm


DEFAULT_CAM = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

# print(DEFAULT_CAM.intrinsic_matrix)
# intrinsic = o3d.io.read_pinhole_camera_intrinsic("intrinsics.json")
intrinsic = DEFAULT_CAM
c_imgs = glob.glob("./input/*")
c_imgs.sort()

d_imgs = glob.glob("./output/*.pfm")
d_imgs.sort()


def clip_image(idepth, threshold=10):
    idepth = idepth - np.amin(idepth[idepth != -np.inf])
    idepth /= np.amax(idepth[idepth != np.inf])

    focal = intrinsic.intrinsic_matrix[0, 0]
    depth = focal / idepth
    depth[depth >= threshold * focal] = np.inf
    return depth

ls = []
for idx in range(len(c_imgs)):
    color = o3d.io.read_image(c_imgs[idx])
    idepth = read_pfm(d_imgs[idx])[0]

    depth = clip_image(idepth)
    print(depth)

    depth = o3d.geometry.Image(depth)

    rgbdi = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth)
    # pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsic)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbdi, intrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    ls.append(pcd)
    # o3d.visualization.draw_geometries([pcd])
o3d.visualization.draw_geometries(ls[:1])
