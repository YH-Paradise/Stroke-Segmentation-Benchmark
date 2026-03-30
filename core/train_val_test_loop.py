import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from evaluation.metrics import measure_dice


def train_model(epoch, train_loader, model, loss_fn, optimizer, device):
    train_loss = []
    size = len(train_loader.dataset)
    model = model.to(device)
    model.train()

    for batch, data in enumerate(train_loader):
        input, gt_label = data[0].to(device), data[1].to(device)
        pred_label = model(input)

        loss = loss_fn(pred_label, gt_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 8 == 7:
            loss, current = loss.item(), (batch + 1) * len(input)
            print(f"[Epoch {epoch + 1}][{current:>5d}/{size:>5d}] loss: {loss:>7f}")
            train_loss.append(loss)

    train_loss_perepoch = np.mean(train_loss)

    return train_loss_perepoch


def val_cal(epoch, val_loader, model, loss_fn, threshold, device):
    val_loss, val_accuracy = [], []
    flag = 0
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for data in val_loader:
            input, gt_label = data[0].to(device), data[1].to(device)

            pred_val = model(input)
            loss = loss_fn(pred_val, gt_label).item()
            val_loss.append(loss)
            flag += 1

            out_cut = np.copy(pred_val.data.cpu().numpy())
            out_cut[out_cut < threshold] = 0.0
            out_cut[out_cut >= threshold] = 1.0
            out_cut = torch.from_numpy(out_cut)

            accuracy = measure_dice(out_cut, gt_label.data.cpu())
            val_accuracy.append(accuracy)

    loss = np.mean(val_loss)
    accuracy = np.mean(val_accuracy)
    print(f"[Epoch {epoch + 1}] Val Error: \n Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {loss:>9f} \n")

    return loss, accuracy


def test_cal(model, test_loader, loss_fn, threshold, device, best_model_path, save_result_npy=False, save_npy_path=''):
    """
    Evaluates the performance of a trained segmentation model on a test dataset.

    Args:
        model (nn.Module): The neural network model to be evaluated.
        test_loader (DataLoader): DataLoader containing the test dataset.
        loss_fn (callable): Loss function used to calculate the error between predicted and ground truth segmentation masks.
        threshold (float): Probability threshold to convert continuous predictions into binary masks.
        device (torch.device): The device (CPU or GPU) to perform evaluation on.
        best_model_path (str): File path to the pre-trained model weights (.pth or .pt).
        save_result_npy (bool, optional): If True, saves the prediction and ground truth arrays. Defaults to False.
        save_npy_path (str, optional): Directory path where the .npy files will be stored. Defaults to ''.

    Returns:
        tuple: A tuple containing:
            - loss (float): The mean loss across the entire test dataset.
            - mean_dice (float): The mean Dice coefficient across the entire test dataset.
    """

    model = model.to(device)

    msg = model.load_state_dict(torch.load(best_model_path), strict = True)
    print(msg)

    model.eval()

    test_loss, test_dice = [], []

    count = 1
    with torch.no_grad():
        for data in test_loader:
            input, gt_label = data[0].to(device), data[1].to(device)

            pred_val = model(input)
            loss = loss_fn(pred_val, gt_label).item()
            test_loss.append(loss)
            pred_label = np.copy(pred_val.data.cpu().numpy())

            if np.squeeze(gt_label.data.cpu().numpy()).max() == 1:
                if save_result_npy:
                    os.makedirs(save_npy_path, exist_ok=True)
                    np.save(f"{save_npy_path}/pred_{count}.npy", pred_label)
                    np.save(f"{save_npy_path}/gt_{count}.npy", gt_label.data.cpu().numpy())
            pred_label[pred_label < threshold] = 0.0
            pred_label[pred_label >= threshold] = 1.0
            pred_label = torch.from_numpy(pred_label)

            dice = measure_dice(pred_label, gt_label.data.cpu())
            test_dice.append(dice)
            count += 1

    loss = np.mean(test_loss)
    mean_dice = np.mean(test_dice)

    print(f"Dice: {mean_dice:>9f}%, Avg loss: {loss:>9f} \n")

    return loss, mean_dice


def visualization(model, test_loader, best_model_path, threshold, device, save_img_path=''):
    """
    Generates and saves visual binary masks from the model's predictions.

    Args:
        model (nn.Module): The neural network model used for prediction.
        test_loader (DataLoader): DataLoader containing the datasets to be predicted and visualized.
            Expected to be 4D tensor (1, Channel, Height, Width).
        best_model_path (str): File path to the saved model state dictionary (.pth or .pt).
        threshold (float): Threshold value to convert continuous model outputs into binary (0 or 1) masks.
        device (torch.device): The device (CPU or GPU) to run the inference on.
        save_img_path (str, optional): Directory path where the generated visualization images will be saved.
            Defaults to ''.

    Returns:
        None

    Note:
        The function assumes the output is a 4D tensor (1, Channel, Height, Width).
    """

    model = model.to(device)

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    with torch.no_grad():
        for j, data in enumerate(test_loader):
            input, gt_label = data[0].to(device), data[1].to(device)
            pred_val = model(input)
            out_cut = np.copy(pred_val.data.cpu().numpy())
            out_cut[out_cut < threshold] = 0.0
            out_cut[out_cut >= threshold] = 1.0
            out_cut = torch.from_numpy(out_cut)

            out_cut = np.squeeze(out_cut, axis=0)
            out_cut = np.squeeze(out_cut, axis=0)
            out_cut = out_cut.numpy()

            for i in range(len(out_cut)):
                os.makedirs(save_img_path, exist_ok=True)
                plt.imsave(f'{save_img_path}/predict_visualization_{j}_{i}.png', out_cut[i], cmap='gray')
