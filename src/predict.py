from gettext import npgettext
from pkgutil import ImpImporter
from train import ResNet18_with_head
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import random
import nibabel as nb
from torchvision import models , transforms

classLabels = [
               "liver",
               "kidney-r",
               "kidney-l",
               "femur-r",
               "femur-l",
               "bladder",
               "heart",
               "lung-r",
               "lung-l",
               "spleen",
               "pancreas",
               ]


def predict_multilabel(num_pred=5):
    data_directory = "/home/upayuryeva/workfolder/test/lits/volumes"
    data = Path(data_directory, "train.csv")
    df = pd.read_csv(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18_with_head(device)

    model.eval()

    df = df[df['person'].isin(set(range(100, 110)))].copy()

    for _ in range(num_pred):
        random_index = random.randint(0, df.shape[0])

        row = df.iloc[random_index]
        curr_img_path = row['img_path']
        img_3d = nb.load(curr_img_path).get_fdata()
        if row['xyz'] == 'x':
            image = img_3d[row['coord'], :, :]
        if row['xyz'] == 'y':
            image = img_3d[:, row['coord'], :]
        if row['xyz'] == 'z':
            image = img_3d[:, :, row['coord']]

        image = np.expand_dims(image, axis=2)
        image = np.repeat(image, 3, axis=2).T
        image = torch.as_tensor(image, dtype=torch.float)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        image = transform(image)
        image = image.to(device)
        preds = model(image.unsqueeze(0))
        preds = (torch.sigmoid(preds).data > 0.5).type(torch.uint8)
        print("PREDICTIONS:")
        print(preds)
        print (','.join([name for pred, name in zip(preds, classLabels)]))
        print("REAL:")
        print(row[[str(el) for el in list(range(1, 12))]])
        # print (','.join([name for pred, name in zip(row[[str(el) for el in list(range(1, 12))]], classLabels)]))

if __name__ == '__main__':
    predict_multilabel()