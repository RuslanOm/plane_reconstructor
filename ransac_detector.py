import open3d as o3d
import cv2
import numpy as np


DEFAULT_CAM = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)


def plane_from_depth_img(map_arr, path="", img=None,
                         threshold=0.2, n_iters=100, color=(255, 0, 0)):
    """Returns params of plane from cloud point"""
    if path:
        img = cv2.imread(path)
    img_3d = o3d.geometry.Image(img)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(img_3d, DEFAULT_CAM)
    plane_model, inliers = pcd.segment_plane(distance_threshold=threshold,
                                             ransac_n=3,
                                             num_iterations=n_iters)

    indeces = np.argwhere(map_arr)[inliers,]
    for i, j in indeces:
        img[i][j] = color

    return plane_model, img

