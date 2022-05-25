import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

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

    # load the dataloaders
    dloadvalid = load_valid_dataset(config)
    logger.info(f"loading {len(dloadvalid) * config.NBATCH:04} samples valid dataset...")

    dloadtrain = load_train_dataset(config)
    logger.info(f"loading {len(dloadtrain) * config.NBATCH:04} samples train dataset...")

    logger.info("datasets.")

    # train on GPU if available, else train on cpu
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}.")

    # load model
    model = timm.create_model(config.MODEL, pretrained=True, num_classes=config.NCLASSES, in_chans=1)
    logger.info("model.")

    # load model to device
    model.to(device)

    # create sigmoid layer
    sigmoid = nn.Sigmoid()
    logger.info('sigmoid layer')

    # load loss
    criterion = nn.BCELoss()
    logger.info("loss.")

    # load loss to device
    criterion.to(device)

    # load optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    logger.info("optimizer.")

    # define scheduler to decrease learning rate with increase in epochs
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: (
            1.0 - step / config.NEPOCHS) if step <= config.NEPOCHS else 0, last_epoch=-1)

    # tensorboard to keep track of loss and accuracy
    writer = SummaryWriter(f"runs/{config.MODEL}")
    logger.info("writer.")

    # start index for epoch
    epoch_start = 0

    model_checkpoints_path = os.path.join(config.CHECKPOINTS_PATH, config.MODEL)

    if not os.path.exists(model_checkpoints_path):
        os.mkdir(model_checkpoints_path)

    model_pretraineds_path = os.path.join(config.PRETRAINEDS_PATH, config.MODEL)

    if not os.path.exists(model_pretraineds_path):
        os.mkdir(model_pretraineds_path)

    # check if there are any checkpoints
    checkpoint_path = best_or_last_checkpoint(path=model_checkpoints_path)

    if checkpoint_path:

        checkpoint = load_checkpoint(path=checkpoint_path)

        # load state dicts of models and optimizers
        optimizer.load_state_dict(checkpoint["optimizer"])
        model.load_state_dict(checkpoint["model"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        epoch_start = checkpoint["epoch"]
        accuracy_valid_best = checkpoint["best_accuracy"]

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: (
                    1.0 - step / config.NEPOCHS) if step <= config.NEPOCHS else 0,
                                                last_epoch=epoch_start)
        logger.info("scheduler.")

        # load and set train mode
        model.train()

        logger.info(f"checkpoint {checkpoint_path} loaded. Start Epoch: {epoch_start}.\n")

    else:

        # define scheduler to decrease learning rate with increase in epochs
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: (
                    1.0 - step / config.NEPOCHS) if step <= config.NEPOCHS else 0, last_epoch=-1)
        logger.info("scheduler.")

        accuracy_valid_best = 0.0
        logger.info("no checkpoint.\n")

    for epoch in range(epoch_start, config.NEPOCHS):

        # ----------------------------------------
        # Train
        # ----------------------------------------

        model.train()

        loss_train          = 0.0
        accuracy_train      = 0.0
        accuracy_batch_best = 0.0

        idx = 0

        for inputs, targets in tqdm(dloadtrain):

            # tensors to device
            inputs, targets = inputs.to(device, dtype=torch.float), targets.float().to(device)
            
            # reset the gradients of all optimized variables
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)

            # scores to sigmoid outputs
            outputs = sigmoid(outputs)

            # calculate the batch loss
            loss    = criterion(outputs, targets)

            # backward pass
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()

            # update training loss
            loss_train += loss.item()

            # update train accuracy
            acc = accuracy(outputs.detach().cpu().numpy(), targets.cpu().numpy())
            accuracy_train += acc # acc[0].item()

            if acc > accuracy_batch_best:
                accuracy_batch_best = acc

            if idx % config.BATCH_LOG_INTERVAL == 0:
                # log train and validation results
                log = f"[{config.MODEL}] | [lr: {optimizer.param_groups[0]['lr']}] | Epoch[{epoch:03}/{config.NEPOCHS:03}]"
                log += f"Batch[{idx:03}/{len(dloadtrain):03}]: "

                log += f"[Train]"
                log += f"[loss: {loss.item():.3f} | "
                log += f"accuracy : {acc:.3f}] | "

                log += f"[best accuracy : {accuracy_batch_best:.3f}] | "
                log += f"[best valid accuracy: {accuracy_valid_best:.3f}]"

                # display log
                print(log)

            if acc > accuracy_batch_best:
                logger.info(log)

            idx += 1

        else:

            # ----------------------------------------
            # Validation
            # ----------------------------------------

            model.eval()

            loss_valid = 0.0
            accuracy_valid = 0.0

            print("\nValidation...")

            with torch.no_grad():

                for inputs, targets in tqdm(dloadvalid):
                    # move tensors to device
                    inputs, targets = inputs.to(device, dtype=torch.float), targets.float().to(device)

                    # forward pass
                    outputs = model(inputs)

                    # sigmoid layer
                    outputs = sigmoid(outputs)

                    # calculate the batch loss
                    loss    = criterion(outputs, targets)

                    # update average validation loss
                    loss_valid += loss.item()

                    # update average validation accuracy
                    acc = accuracy(outputs.detach().cpu().numpy(), targets.cpu().numpy())
                    accuracy_valid += acc # acc[0].item()

            print("done.")

        # update learning rate
        scheduler.step()

        # obtain average accuracy and loss values

        # average loss
        loss_valid /= len(dloadvalid)
        loss_train /= len(dloadtrain)

        # average accuracy
        accuracy_valid /= len(dloadvalid)
        accuracy_train /= len(dloadtrain)

        # log train and validation results
        log = f"\n[{config.MODEL}] | [lr: {optimizer.param_groups[0]['lr']}] | Epoch [{epoch:03}/{config.NEPOCHS:03}]: "

        log += "[Train]"
        log += f"[loss: {loss_train:.3f} | "
        log += f"accuracy : {accuracy_train:.3f}] - "

        log += "[Valid]"
        log += f"[loss: {loss_valid:.3f} | "
        log += f"accuracy : {accuracy_valid:.3f}]\n"

        # display log
        print(log)

        # log loss and accuracy to tensorboard
        writer.add_scalar('Train/Loss', loss_train, epoch)
        writer.add_scalar('Train/Accuracy', accuracy_train, epoch)

        # log loss and accuracy to tensorboard
        writer.add_scalar('Valid/Loss', loss_valid, epoch)
        writer.add_scalar('Valid/Accuracy', accuracy_valid, epoch)

        # save a checkpoint
        if epoch % config.CHECKPOINT_SAVE_INTERVAL == 0:
    
            checkpoint = {

                "epoch"             : epoch,
                "optimizer"         : optimizer,
                "model"             : model,
                "scheduler"         : scheduler,
                "best_accuracy"     : accuracy_valid_best

            }

            name = f"checkpoint_{epoch + 1}.pth"
            save_to = os.path.join(model_checkpoints_path, name)

            save_checkpoint(checkpoint=checkpoint, path=save_to)
            print(f"Checkpoint {name} saved.")

        if accuracy_valid > accuracy_valid_best:
            accuracy_valid_best = accuracy_valid

            checkpoint = {

                "epoch"         : epoch,
                "optimizer"     : optimizer,
                "model"         : model,
                "scheduler"     : scheduler,
                "best_accuracy" : accuracy_valid_best

            }

            name = f"checkpoint_best.pth"
            save_to = os.path.join(model_checkpoints_path, name)

            save_checkpoint(checkpoint=checkpoint, path=save_to)
            print(f"Checkpoint {name} saved.\n")

            torch.onnx.export(
                model,
                torch.randn((1, 1, config.IMAGE_SIZE, config.IMAGE_SIZE),
                             device=inputs.device, requires_grad=True),
                f"./pretraineds/{config.MODEL}/OnnxFloat32.onnx",
                export_params=True,
                opset_version=9,  # the ONNX version to export the model to
                do_constant_folding=True,
                input_names=['input'],  # the model's input names
                output_names=['output'],  # the model's output names
            )

            print("Best model exported as float32 format.\n")
