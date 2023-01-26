#!/usr/bin/env python
# -*- coding: utf-8 -*-
from model import YoloNet
from dataset_params import dataset_params
import os
import numpy as np
DIR = os.path.relpath(os.path.dirname(os.path.realpath(__file__)))
from PIL import Image
import torch
import cv2


def main():
    dataset = "bird_dataset"
    dsp = dataset_params[dataset]
    model = YoloNet(dataset_config=dsp)
    model.load_state_dict(torch.load(f"{DIR}/bird_detection.pth",map_location=torch.device('cpu'))["state_dict"])
    model.eval()
    image = Image.open("./bird.jpeg")
    processed_img = image.resize((dsp["img_w"], dsp["img_h"]), Image.ANTIALIAS)
    processed_img = np.asarray(processed_img)#[:,:,:3]
    processed_img = processed_img / 255.0
    input_x = processed_img.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
    print(input_x.shape)
    output = model(torch.Tensor(input_x))
    print(output.shape)

    

    

if __name__ == "__main__":
    main()
