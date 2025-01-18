import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import timm


def collate_fn(data):
    images, labels = [], []
    for batch in range(0, len(data)):
        images.append(data[batch][0])
        labels.append(data[batch][1])
    data_copy = (np.array(images), np.array(labels))
    return data_copy


class CXR(Dataset):
    def __init__(self, data, portion=1, mode="train"):
        self.data = data
        if mode == "train":
            self.data = self.data[: int(12800 * portion / 100)]
        else:
            self.data = self.data[12800:14848]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return np.array(self.data[idx]["image_npy"]), np.array(
            self.data[idx]["label"], dtype=float
        )


def compute_metrics(predicted, labels):
    correct_predictions = np.all(predicted == labels)
    num_correct = np.sum(correct_predictions)
    num_total = predicted.shape[0]
    accuracy_strict = num_correct / num_total
    TP = np.sum((predicted == 1) & (labels == 1))
    FP = np.sum((predicted == 1) & (labels == 0))
    TN = np.sum((predicted == 0) & (labels == 0))
    FN = np.sum((predicted == 0) & (labels == 1))

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return (
        accuracy_strict,
        accuracy,
        precision,
        recall,
        f1_score,
        TPR,
        FPR,
        TP,
        FP,
        TN,
        FN,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a classifier")
    parser.add_argument("--dataset", type=str, help="数据集", default="pneumonia")
    parser.add_argument("--backbone", type=str, help="视觉编码器", default="resnet")
    parser.add_argument("--gpu", type=int, help="gpu序号", default=0)
    parser.add_argument("--portion", type=int, help="微调数据比例", default=1)
    parser.add_argument("--pretrain", type=bool, help="微调数据比例", default=False)
    args = parser.parse_args()
    return args


args = parse_args()
gpu = args.gpu
device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
dataset = args.dataset
portion = args.portion
pretrain = args.pretrain


if dataset == "chest14":
    datas = np.load("~/data/chest14_train.npy", allow_pickle=True)
elif dataset == "pneumonia":
    datas = np.load("~/data/pneumonia_train.npy", allow_pickle=True)
elif dataset == "vinbigdata":
    datas = np.load("~/data/vinbigdata_train.npy", allow_pickle=True)
train_dataset = CXR(datas, portion)
print(train_dataset.__len__())
train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    collate_fn=collate_fn,
    drop_last=True,
)
eval_dataset = CXR(datas, mode="eval")
print(eval_dataset.__len__())
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    collate_fn=collate_fn,
    drop_last=True,
)

if args.backbone == "convnext":
    model = timm.create_model("convnext_base", pretrained=pretrain)
    # 修改第一层卷积以接受单通道输入
    # 原始的第一层为: Conv2d(3, 96, kernel_size=4, stride=4)
    # 修改为: Conv2d(1, 96, kernel_size=4, stride=4)
    model.stem[0] = nn.Conv2d(
        1,
        model.stem[0].out_channels,
        kernel_size=model.stem[0].kernel_size,
        stride=model.stem[0].stride,
        padding=model.stem[0].padding,
        bias=False,
    )
    if dataset == "chest14":
        model.head.fc = nn.Linear(model.head.in_features, 10)
    elif dataset == "pneumonia":
        model.head.fc = nn.Linear(model.head.in_features, 1)
    elif dataset == "vinbigdata":
        model.head.fc = nn.Linear(model.head.in_features, 8)
    model = model.to(device)
elif args.backbone == "resnet":
    model = models.resnet50(weights=pretrain)
    new_conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    new_conv1.weight.data = model.conv1.weight.mean(dim=1, keepdim=True)
    model.conv1 = new_conv1
    num_ftrs = model.fc.in_features
    if dataset == "chest14":
        model.fc = nn.Linear(num_ftrs, 10)
    elif dataset == "pneumonia":
        model.fc = nn.Linear(num_ftrs, 1)
    elif dataset == "vinbigdata":
        model.fc = nn.Linear(num_ftrs, 8)
    model = model.to(device)


if dataset == "chest14":
    criterion = nn.CrossEntropyLoss()
elif dataset == "pneumonia":
    criterion = nn.BCEWithLogitsLoss()
elif dataset == "vinbigdata":
    criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
with torch.cuda.amp.autocast(enabled=True):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs, labels = torch.tensor(inputs), torch.tensor(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        model.eval()
        correct = 0
        total = 0
        threshold = 0.5
        predicted_concat = []
        label_concat = []
        with torch.no_grad():
            for inputs, labels in eval_dataloader:
                label_concat.append(np.array(labels))
                inputs, labels = torch.tensor(inputs), torch.tensor(labels)
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                predicted = (outputs.data > threshold).int()
                predicted_concat.append(predicted.detach().cpu().numpy())

        predicted_concat = np.concatenate(predicted_concat, axis=0)
        label_concat = np.concatenate(label_concat, axis=0)
        metric = compute_metrics(predicted_concat, label_concat)
        print(metric)
