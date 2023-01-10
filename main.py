from argparse import ArgumentParser
import copy
import os
import pickle
import time

import cv2
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def make_parser():
    parser = ArgumentParser("Generate skeleton data")
    parser.add_argument("-i", "--input", type=str, help="input directory")
    parser.add_argument("-o", "--output", type=str, help="output directory")
    return parser


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f'Use device "{device}"')
    weigths = torch.load('./weights/yolov7-w6-pose.pt')
    yolo = weigths['model']
    yolo = yolo.half().to(device)
    yolo.eval()

    input_dir = args.input
    input_names = os.listdir(input_dir)
    for i, name in enumerate(input_names):
        input_path = os.path.join(input_dir, name)
        logger.info(input_path)


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
