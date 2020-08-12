import torch
import argparse
import os
import numpy as np
import time

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.utils import convert_state_dict

torch.backends.cudnn.benchmark = True
import cv2

n_classes = 19
colors = [#[0, 0, 0],
          [128, 64, 128],
          [244, 35, 232],
          [70, 70, 70],
          [102, 102, 156],
          [190, 153, 153],
          [153, 153, 153],
          [250, 170, 30],
          [220, 220, 0],
          [107, 142, 35],
          [152, 251, 152],
          [0, 130, 180],
          [220, 20, 60],
          [255, 0, 0],
          [0, 0, 142],
          [0, 0, 70],
          [0, 60, 100],
          [0, 80, 100],
          [0, 0, 230],
          [119, 11, 32],
          ]

class_names = [
    # "unlabelled",
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic_light",
    "traffic_sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]
label_colours = dict(zip(range(19), colors))


def decode_segmap(temp):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


def get_segm_by_classes(temp, class_nums):
    r = np.zeros(temp.shape)
    g = np.zeros(temp.shape)
    b = np.zeros(temp.shape)

    for l in class_nums:
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


def get_class_labels():
    return list(enumerate(class_names))


def init_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data_loader = get_loader(args.loader)
    # loader = data_loader(
    #     root="",
    #     is_transform=True,
    #     img_size=eval(args.size),
    #     test_mode=True
    # )
    # n_classes = loader.n_classes

    # Setup Model
    model = get_model({"arch": "hardnet"}, n_classes)
    state = convert_state_dict(torch.load(args["model_path"], map_location=device)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    return device, model


def test(args):
    device, model = init_model(args)
    proc_size = (540, 960) if "size" not in args else eval(args["size"])

    if os.path.isfile(args.input):
        img_raw, decoded, _, _ = process_img(args["input"], proc_size, device, model)
        # blend = np.concatenate((img_raw, decoded), axis=1)
        out_path = os.path.join(args["output"], os.path.basename(args["input"]))
        cv2.imwrite("test.png", decoded * 255)
        # cv2.imwrite(out_path, decoded)

    elif os.path.isdir(args.input):
        print("Process all image inside : {}".format(args["input"]))

        for img_file in os.listdir(args.input):
            _, ext = os.path.splitext(os.path.basename((img_file)))
            if ext not in [".png", ".jpg"]:
                continue
            img_path = os.path.join(args["input"], img_file)

            img, decoded, _, _ = process_img(img_path, proc_size, device, model)
            # blend = np.concatenate((img, decoded), axis=1)
            out_path = os.path.join(args["output"], os.path.basename(img_file))
            cv2.imwrite(out_path, decoded * 255)


def process_img(img_path, size, device, model):
    # print("Read Input Image from : {}".format(img_path))

    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (size[1], size[0]))  # uint8 with RGB mode
    img = img_resized.astype(np.float16)

    # norm
    value_scale = 255
    mean = [0.406, 0.456, 0.485]
    mean = [item * value_scale for item in mean]
    std = [0.225, 0.224, 0.229]
    std = [item * value_scale for item in std]
    img = (img - mean) / std

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    images = img.to(device)
    start = time.time()
    outputs = model(images)
    procc_time = time.time() - start
    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
    decoded = decode_segmap(pred)

    return img_resized, decoded, pred,  procc_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model",
    )

    # parser.add_argument(
    #     "--loader",
    #     type=str,
    #     default="cityscapes",
    #     help="Loader type",
    # )

    parser.add_argument(
        "--size",
        type=str,
        default="540,960",
        help="Inference size",
    )

    parser.add_argument(
        "--input", nargs="?", type=str, default=None, help="Path of the input image/ directory"
    )
    parser.add_argument(
        "--output", nargs="?", type=str, default="./", help="Path of the output directory"
    )
    args = parser.parse_args()
    test(args)
