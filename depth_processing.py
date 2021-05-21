from models.vit_estimator import VITEstimator
from models.midas_estimator import MIDASEstimator
from models import vit
import glob
import os
import cv2


model = VITEstimator(model_path='models/vit/weights/dpt_large-midas-2f21e586.pt')
# model = MIDASEstimator()
input_path = 'input/'
output_path = 'output/'
img_names = glob.glob(os.path.join(input_path, "*"))
num_images = len(img_names)

os.makedirs(output_path, exist_ok=True)

print("start processing")
for ind, img_name in enumerate(img_names):
    if os.path.isdir(img_name):
        continue

    print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))
    if isinstance(model, VITEstimator):
        img = vit.util.io.read_image(img_name)
        model.estimate(img, use_file=output_path + os.path.basename(img_name).split('.')[0] + 'vit' + '.pfm')
    else:
        img = cv2.imread(img_name)
        # img = cv2.resize(img, (640, 480))
        model.estimate(img, use_file=output_path + os.path.basename(img_name).split('.')[0] + 'midas' + '.pfm')
