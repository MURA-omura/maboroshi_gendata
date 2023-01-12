from argparse import ArgumentParser
import json
import os

import cv2
import numpy as np
import torch
from torchvision import transforms
import tqdm

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint


def make_parser():
    parser = ArgumentParser("Generate skeleton data")
    parser.add_argument("-i", "--input", type=str, help="input directory")
    parser.add_argument("-o", "--output", type=str, help="output directory")
    return parser


def main(args):
    os.makedirs(os.path.join(args.output, "maboroshi"), exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Use device "{device}"')
    transform = transforms.ToTensor()

    weigths = torch.load("./weights/yolov7-w6-pose.pt")
    yolo = weigths["model"]
    yolo = yolo.half().to(device)
    yolo.eval()

    input_dir = args.input
    label_dict = dict()

    for label in ["a", "b", "c"]:
        input_names = os.listdir(os.path.join(input_dir, label))
        input_names.remove(".gitignore")

        for name in input_names:
            input_path = os.path.join(os.path.join(input_dir, label), name)
            cap = cv2.VideoCapture(input_path)

            pose_list = [] # 接点座標のリスト

            with tqdm.tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), unit='frame') as pbar:
                pbar.set_description(input_path)
                while True:
                    ret, orig_image = cap.read()
                    if not ret:
                        break
                    image = letterbox(orig_image, 960, stride=64, auto=True)[0]
                    H, W, _ = image.shape

                    image = transform(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output, _ = yolo(image.half())
                    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=yolo.yaml['nc'], nkpt=yolo.yaml['nkpt'], kpt_label=True)
                    with torch.no_grad():
                        output = output_to_keypoint(output)
                    if(len(output) == 0):
                        output = np.zeros((1, 58))

                    multi_pose = [
                        output[:, 7+3*0:10+3*0],
                        (output[:, 7+3*6:10+3*6] + output[:, 7+3*5:10+3*5]) / 2,
                        output[:, 7+3*6:10+3*6],
                        output[:, 7+3*8:10+3*8],
                        output[:, 7+3*10:10+3*10],
                        output[:, 7+3*5:10+3*5],
                        output[:, 7+3*7:10+3*7],
                        output[:, 7+3*9:10+3*9],
                        output[:, 7+3*12:10+3*12],
                        output[:, 7+3*14:10+3*14],
                        output[:, 7+3*16:10+3*16],
                        output[:, 7+3*11:10+3*11],
                        output[:, 7+3*13:10+3*13],
                        output[:, 7+3*15:10+3*15],
                        output[:, 7+3*2:10+3*2],
                        output[:, 7+3*1:10+3*1],
                        output[:, 7+3*4:10+3*4],
                        output[:, 7+3*3:10+3*3],
                    ]
                    multi_pose = np.stack(multi_pose, axis=1)

                    # normalization
                    multi_pose[:, :, 0] = multi_pose[:, :, 0] / W
                    multi_pose[:, :, 1] = multi_pose[:, :, 1] / H
                    multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5
                    multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
                    multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0

                    pose_list.append(multi_pose)
                    pbar.update(1)

            train_data = [
                {
                    "frame_index": i + 1,
                    "skeleton": [
                        {
                            "pose": np.round(p[:, 0:2], 4).reshape(-1).tolist(),
                            "score": np.round(p[:, 2], 4).tolist()
                        }
                        for p in pose
                    ]
                }
                for i, pose in enumerate(pose_list)
            ]

            for i, n in enumerate(range(299, len(train_data), 150)):
                data_dict = {
                    "data": train_data[n-299 : n],
                    "label": label,
                    "label_index": ord(label) - ord("a")
                }
                with open(os.path.join(os.path.join(args.output, "maboroshi"), os.path.splitext(name)[0] + f"_{i}.json"), "w") as f:
                    json.dump(data_dict, f, ensure_ascii=False)

                label_dict.update({
                    os.path.splitext(name)[0]: {
                        "has_skeleton": True,
                        "label": data_dict["label"],
                        "label_index": data_dict["label_index"]
                    }
                })
            cap.release()

    with open(os.path.join(args.output, "maborosi_label.json"), "w") as f:
        json.dump(label_dict, f, ensure_ascii=False)


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
