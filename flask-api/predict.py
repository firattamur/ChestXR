import json
from utils import get_model, process_image


model = get_model()
class_map = json.load(open("../dataset/class_map.json"))


def predict_image(image_bytes: bytes):

    try:

        input  = process_image(image_bytes=image_bytes)
        output = model(input)

    except Exception:

        return 0, 'error'

    _, y_hat = output.max(1)
    predicted_idx = str(y_hat.item())

    return class_map[predicted_idx]