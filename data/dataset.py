import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


def brain_dataset_preparation(path): # dataframe 으로 file path 뽑기
    dwi, gt, mask, adc = [], [], [], []

    set = pd.read_csv(path)

    for i in range(len(set)):
        dwi.append(set.iloc[i] + "/dwi.npy")
        adc.append(set.iloc[i] + "/adc.npy")
        gt.append(set.iloc[i] + "/gt.npy")
        mask.append(set.iloc[i] + "/final_mask.npy")

    df = pd.DataFrame({"dwi": dwi,
                       "adc": adc,
                       "gt": gt,
                       "mask": mask})
    return df


class BrainDWIDataset(Dataset):
    def __init__(self, df, transforms, is_both=False):
        self.df = df
        self.transforms = transforms
        self.is_both = is_both

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        dwi = np.load(self.df.iloc[idx, 0][0])

        mask = np.load(self.df.iloc[idx, 2][0])
        bin_mask = np.load(self.df.iloc[idx, 3][0])

        dwi = np.array(dwi).astype(np.float32)

        mask = np.array(mask).astype(np.float32)

        dwi *= bin_mask

        mask *= bin_mask

        dwi = dwi.unsqueeze(0).unsqueeze(0)
        dwi = F.interpolate(dwi, size=(32, 256, 256), mode='nearest')
        dwi = dwi.squeeze(0).squeeze(0)

        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask, size=(32, 256, 256), mode='nearest')
        mask = mask.squeeze(0).squeeze(0)

        image = dwi

        if self.is_both:
            adc = np.load(self.df.iloc[idx, 1][0])
            adc = np.array(adc).astype(np.float32)
            adc *= bin_mask

            adc = adc.unsqueeze(0).unsqueeze(0)
            adc = F.interpolate(adc, size=(32, 256, 256), mode='nearest')
            adc = adc.squeeze(0).squeeze(0)

            image = torch.stack((dwi, adc), dim = 0) #dwi adc concat

        if image.max() != 0:
            image = (image - image.min()) / image.max()

        return image, mask


def data_transform():
    result = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(256),
        torchvision.transforms.ToTensor()
    ])
    return result


def dataloading(df, shuffle):
    dataset = BrainDWIDataset(df=df, transforms=data_transform())
    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
    return dataloader
