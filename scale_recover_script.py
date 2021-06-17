import open3d as o3d
import numpy as np
import glob
import torch
import matplotlib.pyplot as plt
import cv2

from models.vit.util.io import read_pfm
from models.scale_model import ScaleRecovery


def clip_image(idepth, threshold=10):
    idepth = idepth - np.amin(idepth[idepth != -np.inf])
    idepth /= np.amax(idepth[idepth != np.inf])

    focal = intrinsic.intrinsic_matrix[0, 0]
    depth = focal / idepth
    depth[depth >= threshold * focal] = np.inf
    return depth


cam_height = torch.tensor([2.4])

K = np.array([[583, 0, 320, 0],
              [0, 579.4112, 240, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]], dtype=np.float32)

scale_recovery = ScaleRecovery(1, 480, 640)


DEFAULT_CAM = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

# print(DEFAULT_CAM.intrinsic_matrix)
# intrinsic = o3d.io.read_pinhole_camera_intrinsic("intrinsics_google.json")
intrinsic = DEFAULT_CAM
c_imgs = glob.glob("./input/*")
c_imgs.sort()

d_imgs = glob.glob("tmp_data/output/*.pfm")
d_imgs.sort()

idepth = read_pfm(d_imgs[0])[0]

depth = clip_image(idepth)

depth = torch.tensor([depth])
print(depth.shape)

tensor_K = K.copy()
tensor_K = torch.from_numpy(tensor_K).unsqueeze(0)

scale = scale_recovery(depth, tensor_K, cam_height)
ten = (scale * depth)[0].numpy()
print(np.max(ten[ten != np.inf]), np.min(ten[ten != -np.inf]))
