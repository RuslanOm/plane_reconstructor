from flask import Flask, request, send_file
from flask.json import jsonify
import matplotlib.pyplot as plt
from PIL import Image
import base64
import time
import io

import cv2
import torch
from tqdm import tqdm_notebook

from models.hardnet_segm import HardNetSegm
from models.bts_estimator import BTSEstimator
from plane_detector import PlaneDetector
from models.midas_estimator import MIDASEstimator
from ransac_detector import *
from process_utils import *

print("Initiating model ...")
hardnet_path = "models_pretrained/hardnet_alter_checkpoint.pkl"

depth_model = MIDASEstimator()
segm_model = HardNetSegm(hardnet_path)

model = PlaneDetector(depth_model, segm_model)
print("Model loaded")
app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def hello_world():
    file = request.files['media']
    jpg_as_np = np.frombuffer(file.read(), dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)
    start = time.time()
    res = get_full_cycle(model, img)
    print(f"Time elapsed: {time.time() - start}")
    answer = Image.fromarray(res[0]).tobytes()
    return {"img": base64.encodebytes(answer).decode('utf-8')}


app.run('0.0.0.0', port=5000, debug=True)
