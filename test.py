import timm
from tqdm import tqdm
from utils.utils_train import *
from utils.utils_dataset import *
from utils.utils_commandline import *

if __name__ == '__main__':

    # load configurations
    config = load_config()

    # create logger
    logger = get_logger(config.MODEL)

    # load the data loaders
    dloadtest  = load_test_dataset(config)
    logger.info(f"loading {len(dloadtest)}  samples test  dataset...")

    # train on GPU if available, else train on cpu
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}.")

    # load model
    model = timm.create_model(config.MODEL, pretrained=True, num_classes=config.NCLASSES)
    logger.info("model.")

    model_checkpoints_path = os.path.join(config.CHECKPOINTS_PATH, config.MODEL)

    # check if there are any checkpoints
    checkpoint_path = best_or_last_checkpoint(path=model_checkpoints_path)

    checkpoint = load_checkpoint(path=checkpoint_path)
    model.load_state_dict(checkpoint["efficientnet"])

    # ----------------------------------------
    # Test
    # ----------------------------------------

    model.eval()
    model.to(device)

    accuracy_test = 0.0

    with torch.no_grad():

        for inputs, targets in tqdm(dloadtest):
            # move tensors to device
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)

            # obtain the outputs from the model
            outputs = model(inputs)

            # obtain accuracy
            acc = accuracy(outputs, targets)
            accuracy_test += acc[0].item()

    accuracy_test /= len(dloadtest)

    print(f"Test Accuracy: {accuracy_test:.5f}")

