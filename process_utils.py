import cv2
import torch
from tqdm import tqdm_notebook

from models.bts_estimator import BTSEstimator
from plane_detector import PlaneDetector
from models.midas_estimator import MIDASEstimator
from ransac_detector import *

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors.nearest_centroid import NearestCentroid
from collections import defaultdict
from ransac_detector import *
import scipy.spatial.distance as ds


import numpy as np
from functools import partial
import copy
from sklearn.metrics import pairwise_distances

N = 640 * 480


def straight_pcd(map_arr, depth_map, plane_model):
    a, b, c, d = plane_model
    img = o3d.geometry.Image(np.float32(map_arr * depth_map))
    result = o3d.geometry.PointCloud.create_from_depth_image(img, DEFAULT_CAM)
    result.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    arr = np.asarray(result.points)
    new_points = np.array([[x, y, (-a * x - b * y - d) / c] for x, y, z in arr])
    vec = o3d.utility.Vector3dVector(new_points)
    result.points = vec
    return result


def thr_func(q):
    return partial(np.quantile, q=q)


def _distance_func(v1, v2, th=0.001):
    if abs(v1[-1] - v2[-1]) > th:
        return 1.0
    return ds.cosine(v1[:-1], v2[:-1])


def dist(func):
    return partial(pairwise_distances, metric=func)


def get_plane_img(img, ls_map_arrs):
    """
    Function to plot detected planes on picture
    :param img:
    :param ls_map_arrs:
    :return:
    """
    res = copy.deepcopy(img)
    for map_arr in ls_map_arrs:
        color = [random.randint(1, 255) for _ in range(3)]
        #         print(map_arr)
        ind = np.where(map_arr)
        #         print(ind)

        res[ind[0], ind[1], :] = color
    return res


clf = NearestCentroid(metric='cosine')

clust = AgglomerativeClustering(n_clusters=None,
                                affinity=dist(_distance_func),
                                linkage="average",
                                compute_full_tree=True,
                                distance_threshold=0.2)


def merge_planes(ls_vectors_planes, ls_maps_planes, clust=clust, clf=clf):
    if len(ls_vectors_planes) < 2:
        return ls_vectors_planes, ls_maps_planes
    preds = clust.fit_predict(ls_vectors_planes)
    if len(set(preds)) < 2:
        return [np.mean(ls_vectors_planes, axis=0)], [
            reduce(np.logical_or, ls_maps_planes, np.zeros(ls_maps_planes[0].shape))]
    clf.fit(ls_vectors_planes, preds)
    centroids = {cls: vect for cls, vect in zip(clf.classes_, clf.centroids_)}
    dd = defaultdict(list)
    for cls, map_arr in zip(preds, ls_maps_planes):
        dd[cls].append(map_arr)
    planes = []
    vectors = []
    for item in dd:
        res_map = reduce(np.logical_or, dd[item], np.zeros(dd[item][0].shape))
        planes.append(res_map)
        vectors.append(centroids[item])
    return vectors, planes


kernels = [np.ones((150, 20), np.uint8), np.ones((20, 150), np.uint8)]


def normal_depth(depth_map):
    vect_func = np.vectorize(lambda x: 1 / x if x > 0 else x)
    res = vect_func(depth_map)
    return res


def get_full_cycle(model, img, classes=[{0, 1}]):
    img = cv2.resize(img, (640, 480))
    depth_map, seg_map, ls_map_arr = model.get_segmented_depth(img, classes)
    ls_right_depths = []
    funcs = [thr_func(0.95), np.mean]
    n_depth = normal_depth(depth_map)
    for item, f_th in zip(ls_map_arr, funcs):
        if item.sum():
            tmp, _ = crop_depth_map(depth_map * item, f_th)
            ls_right_depths.extend(get_connected_components(tmp, threshold=0.02))

    ls_maps_planes = []
    ls_vectors_planes = []
    for dd, map_arr in ls_right_depths:
        if map_arr.sum() >= int(0.1 * N):
            result = extract_planes_con(map_arr, dd, map_arr.sum(), 0.1)
            ls_maps_planes.extend([arr for arr, _ in result])
            ls_vectors_planes.extend([arr for _, arr in result])
    ls_maps_planes = [close_map(item, kernels=kernels) for item in ls_maps_planes]

    #     print(ls_maps_planes)
    ls_vectors_planes, ls_maps_planes = merge_planes(ls_vectors_planes, ls_maps_planes)
    ls_maps_planes = [close_map(item, kernels=kernels) for item in ls_maps_planes]
    #     print(ls_vectors_planes)

    # list of initial pcd
    ls_init_pcd = []
    for item in ls_maps_planes:
        tmp = o3d.geometry.Image(np.float32(n_depth * item))
        pcd = o3d.geometry.PointCloud.create_from_depth_image(tmp, DEFAULT_CAM)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        ls_init_pcd.append(pcd)

    # list of straihgt pcd
    ls_straight_pcd = []
    for item, (a, b, c, d) in zip(ls_maps_planes, ls_vectors_planes):
        ls_straight_pcd.append(straight_pcd(item, n_depth, (a, b, c, d)))

    sqew = []
    loss = []
    for x, y in zip(ls_init_pcd, ls_straight_pcd):
        loss.append(loss_metric(x, y))
        sqew.append(abs(np.max(np.asarray(x.points)) - np.min(np.asarray(x.points))) * 1000)
    res_img = get_plane_img(img, ls_maps_planes)
    return res_img, ls_vectors_planes, ls_maps_planes, loss, sqew
