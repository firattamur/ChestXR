import io
import json
import numpy as np
import torch.nn as nn
from PIL import Image
from flaskapi.utils import get_model, process_image, save_grad_cam


def predict_image(config, image_bytes: bytes):

    try:

        model     = get_model(config)
        class_map = json.load(open("../dataset/class_map.json"))

        input  = process_image(config, image_bytes=image_bytes)
        output = nn.Sigmoid()(model(input))

        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        save_grad_cam(config, model, input, image, 0, False)

        _, y_hat = output.max(1)
        predicted_idx = str(y_hat.item())

        print(predicted_idx)

    except Exception as e:
        print(e)
        return 0, 'error'

    return class_map[predicted_idx], class_map[predicted_idx]

