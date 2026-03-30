import random
import torch
import wandb
import argparse

import numpy as np

from core.optimization import optimizer_fc
from core.select_model import model_configuration
from core.train_val_test_loop import train_model, val_cal
from data.brain_lesion_npy_preparation import brain_dataset_preparation, dataloading

seed = 42
random.seed(seed)
np.random.seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def DWI_segmentation_benchmark_train_n_validation(args):
    device = (f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu")

    train_df = brain_dataset_preparation(args.dataset_csv_root_dir + "/trainset.csv")
    val_df = brain_dataset_preparation(args.dataset_csv_root_dir + "/valset.csv")

    model = model_configuration(args.model_name, device)

    loss_fn, unet_optimizer, scheduler = optimizer_fc(model, args.init_lr)
    train_dataloader = dataloading(train_df, shuffle=True)
    val_dataloader = dataloading(val_df, shuffle=False)
    num_epochs = args.num_epochs

    wandb.login(key = args.wandb_key)
    wandb.init(project=args.wandb_project_name, entity=args.wandb_project_entity)

    best_loss = 1e8
    for currentepoch in range(num_epochs):
        print(f"Epoch {currentepoch + 1} \n ------------------------")
        train_loss_perepoch = train_model(currentepoch, train_dataloader, model, loss_fn, unet_optimizer, device)
        val_loss, val_accuracy = val_cal(currentepoch, val_dataloader, model, loss_fn, 0.3, device)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            torch.save(best_model.state_dict(), f"Best_Model/{args.model_name}/best_model_{currentepoch:03d}.pt")

        wandb.log({"Train_Loss": train_loss_perepoch,
                   "Val_Loss": val_loss,
                   "Val_Accuracy": val_accuracy,
                   "lr": scheduler.optimizer.param_groups[0]["lr"]})

        scheduler.step(val_accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MICCAI 2026 AEGIS ONLY")
    parser.add_argument("--config", type=str, default="./config/3D_SwinUNetR_combined.yaml")

    """ WANDB SETTINGS """
    parser.add_argument("--wandb", action="store_true", help="Run with Weights & Biases logging")
    parser.add_argument("--wandb_key", type=str, default="wandb private key")
    parser.add_argument("--wandb_project_entity", type=str, default="wandb private ID")
    parser.add_argument("--wandb_project_name", type=str, default="DWI segmentation benchmark")

    """ DATASET PREPARATION"""
    parser.add_argument("--dataset_csv_root_dir", type=str, default="./dataset")

    """ DEFAULT EXPERIMENT SETTINGS """
    parser.add_argument("--cuda_num", type=int, default=4)
    parser.add_argument("--init_lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--input_shape", nargs="+", type=int, default=[128, 128, 128])
    parser.add_argument("--lookahead", type=str, default="true", choices=["true", "false"])

    """ MODEL SETTINGS """
    parser.add_argument("--model_name", type=str, default="swin")
    parser.add_argument("--is_AEGIS", type=str, default="true")

    """ INPUT SETTINGS """
    parser.add_argument("--dataset", type=str, default="mamamia")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--in_channel", type=int, default=1)
    parser.add_argument("--out_channel", type=int, default=1)

    args = parser.parse_args()

    DWI_segmentation_benchmark_train_n_validation(args)