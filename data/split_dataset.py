import argparse
import os
import glob
import pandas as pd
import numpy as np
from skimage import measure


def split_dataset(base_dir, csv_path='.'):
    root_dir = base_dir

    year2016, year2017, year2018, year2019, year2020 = [], [], [], [], []

    # The dataset used in this study was collected between 2016 and 2020.
    for sub_dir_path in glob.glob(root_dir + "*"):
        if os.path.isdir(sub_dir_path):
            if '2016' in sub_dir_path:
                year2016.append(sub_dir_path)
            elif '2017' in sub_dir_path:
                year2017.append(sub_dir_path)
            elif '2018' in sub_dir_path:
                year2018.append(sub_dir_path)
            elif '2019' in sub_dir_path:
                year2019.append(sub_dir_path)
            elif '2020' in sub_dir_path:
                year2020.append(sub_dir_path)

    years = [year2016, year2017, year2018, year2019, year2020]

    gt_small, gt_med, gt_big = [], [], []
    cnt_label, lesion_cnt = [], []

    for year in years:
        for idx in range(len(year)):
            bin_mask = np.load(year[idx] + "/final_mask.npy")
            gt = np.load(year[idx] + "/gt.npy")

            gt = np.array(gt).astype(np.uint8)
            gt *= bin_mask

            new_label = measure.label(gt)
            cnt_label.append(new_label.max())

            whole_cnt = np.sum(gt == 1)
            lesion_cnt.append(whole_cnt)

            ratio = whole_cnt / new_label.max()

            # Please ensure that constant values for splitting via ratio depends on the dataset.
            if ratio < 120:
                gt_small.append(year[idx])
            elif 120 < ratio < 800:
                gt_med.append(year[idx])
            else:
                gt_big.append(year[idx])

    small_df = pd.DataFrame(gt_small)
    med_df = pd.DataFrame(gt_med)
    big_df = pd.DataFrame(gt_big)

    train_small = small_df.sample(frac=0.75, random_state=42)
    remain_small = small_df.drop(train_small.index)
    val_small = remain_small.sample(frac=0.15 / (0.15 + 0.1), random_state=42)
    test_small = remain_small.drop(val_small.index)

    train_med = med_df.sample(frac=0.75, random_state=42)
    remain_med = med_df.drop(train_med.index)
    val_med = remain_med.sample(frac=0.15 / (0.15 + 0.1), random_state=42)
    test_med = remain_med.drop(val_med.index)

    train_big = big_df.sample(frac=0.75, random_state=42)
    remain_big = big_df.drop(train_big.index)
    val_big = remain_big.sample(frac=0.15 / (0.15 + 0.1), random_state=42)
    test_big = remain_big.drop(val_big.index)

    trainset = pd.concat([train_small, train_med, train_big])
    valset = pd.concat([val_small, val_med, val_big])
    testset = pd.concat([test_small, test_med, test_big])

    os.makedirs(csv_path, exist_ok=True)
    trainset.to_csv(f"{csv_path}/trainset.csv", index=False)
    valset.to_csv(f"{csv_path}/valset.csv", index=False)
    testset.to_csv(f"{csv_path}/testset.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset split via lesion ratio")
    parser.add_argument("--base_dir", type=str, default="", help="Directory path for dataset")
    parser.add_argument("--csv_path", type=str, default="", help="Directory path for saving CSV files")

    args = parser.parse_args()

    split_dataset(args.base_dir, args.csv_path)
