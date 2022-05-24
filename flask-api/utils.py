import io
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms


def get_model():

    model = models.densenet121(pretrained=True)
    model.eval()

    return model


def process_image(image_bytes: bytes):

    preprocess = transforms.Compose(
                                [
                                    transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]
                                )

    image = Image.open(io.BytesIO(image_bytes))

    return preprocess(image).unsqueeze(0)

