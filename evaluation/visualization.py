import torch
import torch.nn as nn
from core.train_val_test_loop import visualization
from data.brain_lesion_npy_preparation import brain_dataset_preparation, dataloading
# from utils.data_preparation.isles_data_preparation import isles_dataset_preparation, dataloading
from monai.networks.nets import SwinUNETR
from core.optimization import optimizer_fc

root_dir = "/home/compu/YH/DWI"
test_df = brain_dataset_preparation(root_dir + "/testset.csv")
test_small_df = brain_dataset_preparation(root_dir + "/testset_small.csv")
test_med_df = brain_dataset_preparation(root_dir + "/testset_med.csv")
test_big_df = brain_dataset_preparation(root_dir + "/testset_big.csv")

jeju_df = brain_dataset_preparation(root_dir + '/jeju_visual_3_7.csv')
test_dataloader = dataloading(jeju_df, shuffle=False)
#
# root_dir = "/home/compu/YH/DWI/ISLES"
# test_small_df = isles_dataset_preparation(root_dir + "/testset_s.csv")
# test_med_df = isles_dataset_preparation(root_dir + "/testset_m.csv")
# test_big_df = isles_dataset_preparation(root_dir + "/testset_b.csv")



device = (
    "cuda:6"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# model = mobilevit_s()
# model = unet3d()
# model = unet_pp()
#
model = SwinUNETR(
        img_size=(32, 256, 256),
        in_channels=2,
        out_channels=1,
        feature_size=48,
        use_checkpoint=True,
    ).to(device)

model = nn.Sequential(model, nn.Sigmoid())
model.to(device)

loss_fn, _, _ = optimizer_fc(model, 3e-4)

best_model_path = "/home/compu/YH/DWI/Best_Model/SwinunetR/ADC+DWI_BCEDICE_cuda7.pt"
# best_model_path = "/home/compu/YH/DWI/Best_Model/UnetPP/ADC+DWI_BCEDICE_cuda6.pt"
# best_model_path = "/home/compu/YH/DWI/Best_Model/MobileVit_unet3d/DWI_real_datasplit_1.pt"
# loss, mean_dice = test_cal(model, test_dataloader, loss_fn, 0.3, device, best_model_path)
visualization(model, test_dataloader, best_model_path, 0.3, device)
