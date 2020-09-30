import os

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors.nearest_centroid import NearestCentroid
from collections import defaultdict
from tqdm import tqdm
from models.hardnet_segm import HardNetSegm
from models.midas_estimator import MIDASEstimator
from plane_detector import PlaneDetector
from ransac_detector import *
import scipy.spatial.distance as ds
import numpy as np
from functools import partial
import copy
from sklearn.metrics import pairwise_distances


hardnet_path = "models/segmnetator/hardnet70_cityscapes_model_2.pkl"
bts_path = "models/depth_estimator/models/bts_latest"

N = 640 * 480


depth_model = MIDASEstimator()
segm_model = HardNetSegm(hardnet_path)
model = PlaneDetector(depth_model, segm_model)


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
        ind = np.where(map_arr)
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
        return np.mean(ls_vectors_planes, axis=0), [reduce(np.logical_or, ls_maps_planes,
                                                          np.zeros(ls_maps_planes[0].shape))]
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


def _get_full_cycle(img):
    depth_map, seg_map, ls_map_arr = model.get_segmented_depth(img, [{0, 1}, {2}])
    ls_right_depths = []
    funcs = [thr_func(0.95), np.mean]
    for item, f_th in zip(ls_map_arr, funcs):
        tmp, _ = crop_depth_map(depth_map * item, f_th)
        ls_right_depths.extend(get_connected_components(tmp, threshold=0.3))

    ls_maps_planes = []
    ls_vectors_planes = []
    for dd, map_arr in ls_right_depths:
        if map_arr.sum() >= int(0.1 * N):
            result = extract_planes_con(map_arr, dd, map_arr.sum(), 0.2)
            ls_maps_planes.extend([arr for arr, _ in result])
            ls_vectors_planes.extend([arr for _, arr in result])
    ls_maps_planes = [close_map(item, kernels=kernels) for item in ls_maps_planes]

    ls_vectors_planes, ls_maps_planes = merge_planes(ls_vectors_planes, ls_maps_planes)
    ls_maps_planes = [close_map(item, kernels=kernels) for item in ls_maps_planes]

    ans = get_plane_img(img, ls_maps_planes)
    return ans, ls_vectors_planes, ls_maps_planes


if __name__ == "__main__":
    path = "/home/ruslan/Документы/startup/FCHarDNet/data/leftImg8bit/val/frankfurt/"
    files = os.listdir(path)

    images = [cv2.resize(cv2.imread(path + item), (640, 480)) for item in tqdm(files[:50])]
    ans = _get_full_cycle(images[42])
    cv2.imwrite("result.png", ans[0])
    print("Done!...")
