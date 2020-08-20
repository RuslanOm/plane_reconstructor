import open3d as o3d
import numpy as np
import copy
from typing import Tuple


DEFAULT_CAM = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)


def plane_from_depth_img(map_arr: np.ndarray,
                         image: np.ndarray,
                         depth_map: np.ndarray,
                         threshold=0.2,
                         n_iters=100,
                         color=(255, 0, 0)) -> Tuple[np.ndarray[np.float64[4, 1]], np.ndarray]:
    """Returns params of plane from cloud point"""
    img_3d = o3d.geometry.Image(depth_map)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(img_3d, DEFAULT_CAM)
    plane_model, inliers = pcd.segment_plane(distance_threshold=threshold,
                                             ransac_n=3,
                                             num_iterations=n_iters)

    indeces = np.argwhere(map_arr)[inliers,]
    for i, j in indeces:
        image[i][j] = color

    return plane_model, image


def crop_depth_map(depth_map: np.ndarray, threshold=0.3) -> np.ndarray:
    """Crops depth map from above"""
    x, y = depth_map.shape
    result = np.asarray(copy.copy(depth_map))
    for i in range(int(x * threshold) + 1):
        result[i, ] *= 0
    return result


def downsample_pcd(pcd: o3d.geometry.PointCloud, voxel=0.05) -> o3d.geometry.PointCloud:
    """Downsample pointcloud"""
    return pcd.voxel_down_sample(voxel_size=voxel)


def estimate_normals_from_depth_pcd(pcd: o3d.geometry.PointCloud, rad=0.1, max_nn=30) -> np.ndarray:
    """Method for estimating normals from depth pcd"""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=rad, max_nn=max_nn))
    return np.asarray(pcd.normals)


def estimate_plane(depth_map: np.ndarray,
                   is_crop=True,
                   is_down_sample=True,
                   threshold=0.1,
                   n_iters=100,
                   crop_kwargs={},
                   downs_kwargs={}):
    """ Method for estimating one plane from depth_map
    :param depth_map: nd.array of depth per pixel
    :param is_crop: if True then depth_map will be croped from above
    :param is_down_sample: if True then PointCloud will be downsampled
    :param threshold: param for ransac method; defines width of plane
    :param n_iters: param of ransac method;
    :param crop_kwargs: kwargs for crop_depth_map
    :param downs_kwargs: kwargs for downsample_pcd
    :return:
    """
    if is_crop:
        res = crop_depth_map(depth_map, **crop_kwargs)
        img_3d = o3d.geometry.Image(res)
    else:
        img_3d = o3d.geometry.Image(depth_map)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(img_3d, DEFAULT_CAM)
    if is_down_sample:
        pcd = downsample_pcd(pcd, **downs_kwargs)
    plane_model, inliers = pcd.segment_plane(distance_threshold=threshold,
                                             ransac_n=3,
                                             num_iterations=n_iters)

    return plane_model
