# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time
import json
import uuid
import urllib.request
from urllib.parse import urlparse

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

SAMPLE_IMAGE_PATH = "C:/python/face_recognition/examples/img/"

# 因为安卓端APK获取的视频流宽高比为3:4,为了与之一致，所以将宽高比限制为3:4
def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        return False
    else:
        return True


def test(image_path, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    test_speed = 0
    start = time.time()

    # if read image from path
    img = cv2.imread(image_path)
    head, tail = os.path.split(image_path)
    image_name = tail
    image_output = img

    # check image ratio
    result = check_image(img)
    if result is False:
        # resize image ratio to 3/4
        image_output = cv2.resize(img, (int(img.shape[0] * 3/4), img.shape[0]))
        # save resized image
        # format_ori = os.path.splitext(image_name)[-1]
        # resize_image_name = image_name.replace(format_ori, "_resize" + format_ori)
        # cv2.imwrite(SAMPLE_IMAGE_PATH + resize_image_name, image_output)
    
    image_bbox = model_test.get_bbox(image_output)
    prediction = np.zeros((1, 3))
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image_output,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time()-start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    if label == 1:
        result_real = True
        result_message = "Real Face"
        result_score = "{:.2f}".format(value)
        result_text = "RealFace Score: {:.2f}".format(value)
        color = (255, 0, 0)
    else:
        result_real = False
        result_message = "Fake Face"
        result_score = "{:.2f}".format(value)
        result_text = "FakeFace Score: {:.2f}".format(value)
        color = (0, 0, 255)

    data = {
        "real": result_real,
        "score": result_score,
        "image_name": image_name,
        "time": "{:.2f}s".format(test_speed),
        "message": result_message
    }

    json_str = json.dumps(data)
    print(json_str)

    # save result image with rectangel & label based on result
    # cv2.rectangle(
    #     image_output,
    #     (image_bbox[0], image_bbox[1]),
    #     (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
    #     color, 2)
    # cv2.putText(
    #     image_output,
    #     result_text,
    #     (image_bbox[0], image_bbox[1] - 5),
    #     cv2.FONT_HERSHEY_COMPLEX, 0.5*image_output.shape[0]/1024, color)

    # format_ = os.path.splitext(image_name)[-1]
    # result_image_name = image_name.replace(format_, "_result" + format_)
    # cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image_output)

if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="C:/python/face_recognition/face_recognition/resources/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--image_path",
        type=str,
        default="C:/python/face_recognition/examples/img/image_T1.jpg",
        help="image path used to test")
    args = parser.parse_args()
    test(args.image_path, args.model_dir, args.device_id)
