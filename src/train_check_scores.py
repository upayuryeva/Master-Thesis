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

from sklearn.metrics import precision_score, recall_score
from collections import defaultdict

import matplotlib.pyplot as plt

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
    
    # checkpoint = torch.load("/home/upayuryeva/workfolder/test/lits/src/chkpts/epoch-2-acc-0.33")
    # model.load_state_dict(checkpoint["model"])

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

    train_loss_plot = []
    test_loss_plot= []


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
    
            recalls = defaultdict(list)
            precisions = defaultdict(list)
            scores_labels = defaultdict(list)
            scores = [0] * 199

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
                        # print(output)
                        # print(target)
                        loss = criterion(output, target)
                        # print(output[:, 0])
                        # print(output.shape)
                        if phase == "train":
                            loss.backward()

                            optimizer.step()

                        running_loss += loss.item() * data.size(0)

                        for idx_thrs, sigmoid_thr in enumerate(np.arange(0.005, 1, 0.005)):
                          preds = torch.sigmoid(output).data > sigmoid_thr
                          preds = preds.to(torch.float32)
                          for j in range(1, 12):
                            if len(recalls[j]) == 0:
                              recalls[j] = [0] * 199
                              precisions[j] = [0] * 199
                              scores_labels[j] = [0] * 199
                            recalls[j][idx_thrs] = recalls[j][idx_thrs] + (
                                recall_score(
                                    target.to("cpu").to(torch.int).numpy()[:, j-1],
                                    preds.to("cpu").to(torch.int).numpy()[:, j-1],
                                    zero_division=1
                                    )
                                )
                            precisions[j][idx_thrs] = precisions[j][idx_thrs] + (
                                precision_score(
                                    target.to("cpu").to(torch.int).numpy()[:, j-1],
                                    preds.to("cpu").to(torch.int).numpy()[:, j-1],
                                    zero_division=1
                                    )
                                )
                            scores_labels[j][idx_thrs] = scores_labels[j][idx_thrs] + (
                                f1_score(
                                    target.to("cpu").to(torch.int).numpy()[:, j-1],
                                    preds.to("cpu").to(torch.int).numpy()[:, j-1],
                                    zero_division=1
                                    )
                                )

                          scores[idx_thrs] = scores[idx_thrs] + (
                              f1_score(
                                  target.to("cpu").to(torch.int).numpy(),
                                  preds.to("cpu").to(torch.int).numpy(),
                                  average="samples", zero_division=1
                                  )
                              )
                              
                        preds = torch.sigmoid(output).data > 0.75
                        preds = preds.to(torch.float32)
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
                epoch_shape = train_df.shape[0]
                epoch_loss = running_loss / epoch_shape
                train_loss_plot.append(epoch_loss)
                epoch_acc = running_corrects / epoch_shape
                epoch_by_label = score_by_label / epoch_shape
            else:
                epoch_shape = test_df.shape[0]
                epoch_loss = running_loss / epoch_shape
                test_loss_plot.append(epoch_loss)
                epoch_acc = running_corrects / epoch_shape
                epoch_by_label = score_by_label / epoch_shape
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    state = {
                        'epoch': epoch, 
                        'model': model.state_dict(), 
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }
                    torch.save(state, f"/home/upayuryeva/workfolder/test/lits/src/chkpts/epoch-{epoch+13}-acc-{round(best_acc, 2)}")
                    # best_model_wts = copy.deepcopy(model.state_dict())
            fig, axs = plt.subplots(3, 4, figsize=(15, 15))
            for j in range(1, 12):
              #create precision recall curve
              axs[(j-1) // 4 , (j-1) % 4].set(
                  title=f'Precision-Recall Curve {classLabels[j-1]}', ylabel='Precision', xlabel="Recall") 

              axs[(j-1) // 4 , (j-1) % 4].plot(
                  np.array(recalls[j])/df_t["batch_num"].max(), 
                  np.array(precisions[j])/df_t["batch_num"].max(), 
                  color='purple'
                  )
            fig, axs = plt.subplots(3, 4, figsize=(15, 15))
            for j in range(1, 12):
              #create precision recall curve
              axs[(j-1) // 4 , (j-1) % 4].set(
                  title=f'F-1 for {classLabels[j-1]}', ylabel='F1', xlabel="Threshold") 

              axs[(j-1) // 4 , (j-1) % 4].plot(
                  np.arange(0.005, 1, 0.005), 
                  np.array(scores_labels[j])/df_t["batch_num"].max(), 
                  color='purple'
                  )

            

              #display plot
            
            plt.show()

            fig, ax = plt.subplots() 
            ax.plot(np.arange(0.005, 1, 0.005), np.array(scores)/df_t["batch_num"].max(),  color='purple')

            #add axis labels to plot
            ax.set_title('Scores')
            ax.set_ylabel('F-1')
            ax.set_xlabel('Threshold')

            print(np.array(scores)/df_t["batch_num"].max())

            #display plot
            plt.show()

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
            fig, ax = plt.subplots() 
            ax.plot(np.array(train_loss_plot), color='blue')
            ax.plot(np.array(test_loss_plot), color='red')

            #add axis labels to plot
            ax.set_title('Scores')
            ax.set_ylabel('Loss')
            ax.set_xlabel('Epoch')

            #display plot
            plt.show()
            total_norm = 0
            parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
            for p in parameters:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print('Norm grad', total_norm)
            

    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(state["model"])

    torch.save(state, f"/home/upayuryeva/workfolder/test/lits/src/chkpts/epoch-{num_epochs}-acc-{round(best_acc, 2)}")


def train_multilabel():
    data_directory = "/home/upayuryeva/workfolder/test/lits/volumes/"
    data = Path(data_directory, "train.csv")
    df = pd.read_csv(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = ResNet18_with_head(device)

    train_df = df[df['person'].isin(set(range(20, 55)))].copy()
    # train_df = train_df.iloc[:train_df.shape[0] // 2]
    test_df = df[df['person'].isin(set(range(105, 110)))].copy()
    # test_df = test_df.iloc[:test_df.shape[0] // 2]
    # train_df = df[df['person'].isin(set(range(0, 3)))].copy()
    # test_df = df[df['person'].isin(set(range(3, 5)))].copy()

    weight_list = ((train_df.shape[0] - train_df[[str(el) for el in list(range(1, 12))]].sum()) / train_df[[str(el) for el in list(range(1, 12))]].sum()).to_list()

    pos_weights = torch.as_tensor(weight_list, dtype=torch.float, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    batch_size = 512
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=train_df.shape[0], num_warmup_steps=round(train_df.shape[0]*0.05))

    train(model, device, train_df, test_df, batch_size=batch_size, criterion=criterion, optimizer=optimizer, scheduler=scheduler, num_epochs=15)
    # sgdr_partial = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.005 )
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

if __name__ == '__main__':
    train_multilabel()