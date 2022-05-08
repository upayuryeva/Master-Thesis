import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models , transforms
import json
from torch.utils.data import Dataset, DataLoader ,random_split
from PIL import Image
from pathlib import Path
import torch.optim as optim
from torch.optim import lr_scheduler

from tqdm import trange
from tqdm import tqdm
from sklearn.metrics import precision_score,f1_score
import copy

from transformers import get_cosine_schedule_with_warmup

from data_processing import get_df_train_coord

import nibabel as nb
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

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


def create_head(num_features, number_classes, dropout_prob=0.5, activation_func=nn.ReLU):
    features_lst = [num_features, num_features // 2, num_features // 4]
    layers = []
    for in_f, out_f in zip(features_lst[:-1], features_lst[1:]):
        layers.append(nn.Linear(in_f, out_f))
        layers.append(activation_func())
        layers.append(nn.BatchNorm1d(out_f))
        if dropout_prob != 0: layers.append(nn.Dropout(dropout_prob))
    layers.append(nn.Linear(features_lst[-1], number_classes))
    return nn.Sequential(*layers)


def ResNet18_with_head(device):



    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features

    top_head = create_head(num_features, len(classLabels))
    model.fc = top_head

    model = model.to(device)
    
    checkpoint = torch.load("chkpts/epoch-10-acc-0.39")
    model.load_state_dict(checkpoint["model"])

    return model


def get_batch(df, idx, person):

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    df_batch = df[df["batch_num"] == idx]
    curr_img_path = df_batch.iloc[0]['img_path']
    img_3d = nb.load(curr_img_path).get_fdata()
    batch_img = []
    batch_lbl = []
    for i, row in df_batch.iterrows():
        # if row["person"] != person:
        #     print(row["person"])
        #     person = row["person"]
        if row['img_path'] != curr_img_path:
            curr_img_path = row['img_path']
            img_3d = nb.load(curr_img_path).get_fdata()
        if row['xyz'] == 'x':
            image = img_3d[row['coord'], :, :]
        if row['xyz'] == 'y':
            image = img_3d[:, row['coord'], :]
        if row['xyz'] == 'z':
            image = img_3d[:, :, row['coord']]

        label = torch.tensor(row[[str(el) for el in list(range(1, 12))]].tolist(), dtype=torch.float32)

        image = np.expand_dims(image, axis=2)
        image = np.repeat(image, 3, axis=2).T
        image = torch.as_tensor(image, dtype=torch.float)
        image = transform(image)
        batch_img.append(image)
        batch_lbl.append(label)
    yield (
        torch.stack(batch_img),
        torch.stack(batch_lbl)
    )


def train(model, device, train_df, test_df, batch_size, criterion, optimizer, scheduler, num_epochs=10):  # scheduler,

    train_df["batch_num"] = [i // batch_size for i in range(train_df.shape[0])]
    test_df["batch_num"] = [i // batch_size for i in range(test_df.shape[0])]
    person = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in trange(num_epochs, desc="Epochs"):
        result = []
        for phase in ['train', 'val']:
            if phase == "train":  # put the model in training mode
                model.train()
                len_data_train = 0
            else:  # put the model in validation mode
                model.eval()
                len_data_test = 0

            # keep track of training and validation loss
            running_loss = 0.0
            running_corrects = 0.0
            score_by_label = np.zeros(11)
            if phase == 'train':
                df_t = train_df.copy()
            else:
                df_t = test_df.copy()

            for i in trange(df_t["batch_num"].max()):
                batch = get_batch(df_t, i, person)
                if phase == 'train':
                    len_data_train += batch_size
                else:
                    len_data_test += batch_size
                for data, target in batch:
                    # for data, target in tqdm(dataloader[phase]):
                    # load the data and target to respective device

                    data, target = data.to(device), target.to(device)

                    with torch.set_grad_enabled(phase == "train"):
                        optimizer.zero_grad()

                        output = model(data)
                        loss = criterion(output, target)

                        preds = torch.sigmoid(output).data > 0.5
                        preds = preds.to(torch.float32)

                        if phase == "train":
                            loss.backward()

                            optimizer.step()
                        running_loss += loss.item() * data.size(0)
                        score = f1_score(
                            target.to("cpu").to(torch.int).numpy(),
                            preds.to("cpu").to(torch.int).numpy(),
                            average="samples", zero_division=1
                        ) * data.size(0)

                        score_by_label_local = f1_score(
                            target.to("cpu").to(torch.int).numpy(),
                            preds.to("cpu").to(torch.int).numpy(),
                            average=None, zero_division=1
                        ) * data.size(0)
                        running_corrects += score
                        score_by_label = score_by_label + score_by_label_local

                    if phase=="train":
                        scheduler.step()

                    # print(score)

            if phase == 'train':
                epoch_loss = running_loss / train_df.shape[0]
                epoch_acc = running_corrects / train_df.shape[0]
                epoch_by_label = score_by_label / train_df.shape[0]
            else:
                epoch_loss = running_loss / test_df.shape[0]
                epoch_acc = running_corrects / test_df.shape[0]
                epoch_by_label = score_by_label / test_df.shape[0]
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    state = {
                        'epoch': epoch, 
                        'model': model.state_dict(), 
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }
                    # best_model_wts = copy.deepcopy(model.state_dict())
            result.append('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print()
            print(result)
            print()
            # print(display(
            #     pd.DataFrame(
            #         np.array([epoch_by_label]),
            #         columns=classLabels)
            # )
            # )
            print(epoch_by_label)
            print(classLabels)
            total_norm = 0
            parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
            for p in parameters:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print('Norm grad', total_norm)

    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(state["model"])

    torch.save(state, f"chkpts/epoch-{num_epochs}-acc-{round(best_acc, 2)}")


def train_multilabel():
    df = get_df_train_coord()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = ResNet18_with_head(device)

    train_df = df[df['person'].isin(set(range(20, 80)))].copy()
    test_df = df[df['person'].isin(set(range(100, 108)))].copy()
    # train_df = df[df['person'].isin(set(range(0, 3)))].copy()
    # test_df = df[df['person'].isin(set(range(3, 5)))].copy()

    # weight_list = ((train_df.shape[0] - train_df[[str(el) for el in list(range(1, 12))]].sum()) / train_df[[str(el) for el in list(range(1, 12))]].sum()).to_list()

    weight_list = (train_df[[str(el) for el in list(range(1, 12))]].sum() / train_df.shape[0]).to_list()

    pos_weights = torch.as_tensor(weight_list, dtype=torch.float, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    batch_size = 512
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=train_df.shape[0], num_warmup_steps=round(train_df.shape[0]*0.05))

    train(model, device, train_df, test_df, batch_size=batch_size, criterion=criterion, optimizer=optimizer, scheduler=scheduler, num_epochs=10)
    # sgdr_partial = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.005 )
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

if __name__ == '__main__':
    train_multilabel()