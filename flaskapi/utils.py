import io
import os
import timm
from PIL import Image
import torchvision.transforms as transforms
from utils.utils_train import best_or_last_checkpoint, load_checkpoint


def get_model(config):

    # load model
    model = timm.create_model(config.MODEL, pretrained=True, num_classes=config.NCLASSES)

    # checkpoint path
    model_checkpoints_path = os.path.join(config.CHECKPOINTS_PATH, config.MODEL)
    print(model_checkpoints_path)
    # check if there are any checkpoints
    checkpoint_path = best_or_last_checkpoint(path=model_checkpoints_path)

    print(checkpoint_path)
    checkpoint = load_checkpoint(path=checkpoint_path)
    print(checkpoint)

    model.load_state_dict(checkpoint["model"])

    model.eval()

    return model


def process_image(config, image_bytes: bytes):

    preprocess = transforms.Compose(
                                    [
                                        transforms.Resize(config.IMAGE_SIZE + 1),
                                        transforms.CenterCrop(config.IMAGE_SIZE),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]
                                )

    image = Image.open(io.BytesIO(image_bytes))

    return preprocess(image).unsqueeze(0)

