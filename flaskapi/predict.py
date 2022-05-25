import json
from flaskapi.utils import get_model, process_image


def predict_image(config, image_bytes: bytes):

    try:

        model     = get_model(config)
        class_map = json.load(open("../dataset/class_map.json"))

        input  = process_image(config, image_bytes=image_bytes)
        output = model(input)

    except Exception as e:
        print(e)
        return 0, 'error'

    _, y_hat = output.max(1)
    predicted_idx = str(y_hat.item())

    return class_map[predicted_idx]

