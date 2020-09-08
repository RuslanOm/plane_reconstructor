import open3d as o3d
import numpy as np
import copy
from typing import Callable
import cv2
import random

DEFAULT_SHAPE = (640, 480)
DEFAULT_CAM = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)


class Transform:
    def __init__(self, first_block, transformations, last_block):
        """
        This class represents pipeline for processing depth maps
        :param transformations: list of chained functions
        """
        self.first_block = first_block
        self.transformations = transformations
        self.last_block = last_block

    def process_input(self, depth_map):
        """
        Main function to start pipeline
        :param depth_map: input depth map for plane segmentation
        :return: result plane equations and inliers
        """
        tmp = copy.deepcopy(depth_map)
        inp = self.first_block(tmp)
        for item, args, kwargs in self.transformations:
            inp = item(inp, *args, **kwargs)
        answer = self.last_block(inp)
        return answer


def crop_depth_map(depth_map, threshold=np.mean):
    """
    Function to crop pixel with big depth
    :param depth_map: input depth map
    :param threshold: function, float or int
    :return: croped depth map, number of nonzero pixels
    """
    vect_func = np.vectorize(lambda x: 1 / x if x > 0 else x)
    tmp = vect_func(depth_map)
    if isinstance(threshold, Callable):
        arr = tmp.flatten()
        t = threshold(arr[arr > 0])
    else:
        t = threshold
    ans = np.where(tmp < t, tmp, 0)
    return ans, len(np.argwhere(ans))


def get_connected_components(depth_map, threshold=0.3):
    """
    Function to split depth map into separated components
    :param depth_map: input depth map
    :param threshold: float, components, which less then this value, will be skipped
    :return: list of depth maps
    """
    arr = np.uint8(np.where(depth_map > 0, 1, 0))
    count_pixels = arr.sum()
    num, result = cv2.connectedComponents(arr)
    ans = []
    for i in range(1, num):
        tmp = np.where(result == i, 1, 0)
        if tmp.sum() / count_pixels >= threshold:
            ans.append(np.float32(depth_map * tmp))
    return ans


def _update_map_arr(map_arr, ind):
    for i, j in ind:
        map_arr[i][j] = 0


def _get_plane(pcd, th):
    return pcd.segment_plane(distance_threshold=th,
                             ransac_n=3,
                             num_iterations=100)


def _get_pcd(depth_map):
    img_3d = o3d.geometry.Image(depth_map)
    return o3d.geometry.PointCloud.create_from_depth_image(img_3d, DEFAULT_CAM)


def _get_plane_from_pcd(depth_map):
    img_3d = o3d.geometry.Image(np.float32(depth_map))
    pcd = o3d.geometry.PointCloud.create_from_depth_image(img_3d, DEFAULT_CAM)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    arr = np.asarray(pcd.points)
    th = abs(np.max(arr) - np.min(arr)) / 100
    plane_model, inliers = pcd.segment_plane(distance_threshold=th,
                                             ransac_n=3,
                                             num_iterations=100)
    return plane_model, inliers


# recursion
# def extract_planes_rec(map_arr, depth_map, start_size, t, result):
#     curr_map_arr = copy.deepcopy(map_arr)
#     plane_model, inliers = _get_plane_from_pcd(depth_map * curr_map_arr)
#     if len(inliers) < int(start_size * t):
#         return []
#     else:
#         ls = np.argwhere(map_arr)
#         np_ls = np.array(ls)
#         ans = np.zeros((480, 640))
#         for i, j in np_ls[inliers]:
#             curr_map_arr[i][j] = 0
#             ans[i][j] = 1
#         result.append((ans, plane_model))
#         extract_planes_rec(curr_map_arr, depth_map, start_size, t, result)


# consequentive
def extract_planes_con(map_arr, depth_map, start_size, t):
    curr_map_arr = copy.deepcopy(map_arr)
    plane_model, inliers = _get_plane_from_pcd(depth_map * curr_map_arr)

    result = []
    while len(inliers) >= int(start_size * t):
        ls = np.argwhere(curr_map_arr)
        np_ls = np.array(ls)
        ans = np.zeros((480, 640))
        for i, j in np_ls[inliers]:
            curr_map_arr[i][j] = 0
            ans[i][j] = 1
        result.append((ans, plane_model))
        plane_model, inliers = _get_plane_from_pcd(depth_map * curr_map_arr)
    return result


def get_plane_img(img, ls_map_arrs):
    """
    Function to plot detected planes on picture
    :param img:
    :param ls_map_arrs:
    :return:
    """
    res = copy.deepcopy(img)
    for map_arr in ls_map_arrs:
        indeces = np.argwhere(map_arr)
        color = [random.randint(1, 255) for _ in range(3)]
        for i, j in indeces:
            res[i][j] = color
    return res


def plane_from_depth_img(map_arr: np.ndarray,
                         image: np.ndarray,
                         depth_map: np.ndarray,
                         threshold=0.2,
                         n_iters=100,
                         color=(255, 0, 0)):
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
    if len(np.asarray(pcd.points)) >= 3:
        plane_model, inliers = pcd.segment_plane(distance_threshold=threshold,
                                                 ransac_n=3,
                                                 num_iterations=n_iters)

        return plane_model
    return 0, 0, 0, 0
