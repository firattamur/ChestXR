import io
import json
import numpy as np
import torch.nn as nn
from PIL import Image
from flaskapi.utils import get_model, process_image, save_grad_cam


def predict_image(config, image):

    response = {}

    try:

        model     = get_model(config)
        class_map = json.load(open("../dataset/class_map.json"))

        input  = process_image(config, image=image)
        output = nn.Sigmoid()(model(input))

        output_sorted = np.sort(output[0].detach().numpy())

        top1_score = output_sorted[-1]
        top2_score = output_sorted[-2]
        top3_score = output_sorted[-3]

        top1_index = np.where(output[0] == top1_score)[0][0]
        top2_index = np.where(output[0] == top2_score)[0][0]
        top3_index = np.where(output[0] == top3_score)[0][0]

        top1_category = class_map[str(top1_index)]
        top2_category = class_map[str(top2_index)]
        top3_category = class_map[str(top3_index)]

        top1_image = save_grad_cam(config, model, input, image, top1_index, False)
        top2_image = save_grad_cam(config, model, input, image, top2_index, False)
        top3_image = save_grad_cam(config, model, input, image, top3_index, False)

        response = {

            "top1_image"    : top1_image,
            "top1_category" : f"{top1_category} : {top1_score:.2f}",

            "top2_image"    : top2_image,
            "top2_category" : f"{top2_category} : {top2_score:.2f}",

            "top3_image"    : top3_image,
            "top3_category" : f"{top3_category} : {top3_score:.2f}",

        }

    except Exception as e:
        print(e)

    return response

