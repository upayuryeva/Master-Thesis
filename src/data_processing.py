import pandas as pd

import nibabel as nb

from pathlib import Path
from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt
from nilearn import plotting as nlp  # Neiroimage plotting
import transforms3d
from scipy import ndimage as ndi
import nibabel.testing
from collections import defaultdict


def append_label(
        train_info,
        img_data,
        xyz,
        i_xyz,
        coord_1,
        coord_2,
        img_n,
        annot_info,
        img_path
):
    for i in range(0, img_data.shape[i_xyz]):
        train_info["img_path"] .append(img_path)
        train_info['img_name'].append(f"{img_n}_{xyz}_{i}")
        train_info['person'].append(img_n)
        train_info['xyz'].append(xyz)
        train_info['coord'].append(i)
        for label in range(1, 12):
            label = str(label)
            if label in annot_info.keys():
                if (
                        (int(annot_info[label][coord_1]) <= i)
                        and (int(annot_info[label][coord_2]) >= i)
                ):
                    train_info[label].append(1)
                else:
                    train_info[label].append(0)
            else:
                train_info[label].append(0)

    return train_info


def get_df_train_coord():
    # training
    indexes = list(range(0, 131))
    train_info = defaultdict(list)
    data_directory = "directory"

    for img_num in indexes:
        annot_info = {}
        img_vol_path = Path(data_directory, f"volume-{img_num}.nii")
        annot_path = Path(data_directory, f"segmentation-{img_num}.txt")
        with open(annot_path) as f:
            lines = f.readlines()

        for l in lines:
            l = l.split(' ')
            l[-1] = l[-1][:-1]
            annot_info[l[1]] = l[2:]

        img_data = nb.load(img_vol_path)

        train_info = append_label(
          train_info=train_info,
          img_data=img_data,
          xyz='x',
          i_xyz=0,
          coord_1=0,
          coord_2=1,
          img_n=img_num,
          annot_info=annot_info,
          img_path=img_vol_path
        )
        train_info = append_label(
          train_info=train_info,
          img_data=img_data,
          xyz='y',
          i_xyz=1,
          coord_1=2,
          coord_2=3,
          img_n=img_num,
          annot_info=annot_info,
          img_path=img_vol_path
        )
        train_info = append_label(
          train_info=train_info,
          img_data=img_data,
          xyz='z',
          i_xyz=2,
          coord_1=4,
          coord_2=5,
          img_n=img_num,
          annot_info=annot_info,
          img_path=img_vol_path
        )

    train_df = pd.DataFrame(data=train_info)

    train_df.to_csv(Path(data_directory, "train.csv"), index=False)

    return train_df


