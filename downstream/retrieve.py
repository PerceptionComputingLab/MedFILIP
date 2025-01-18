import numpy as np
import torch
from models import VLM


device = torch.device("cuda:{}".format(4) if torch.cuda.is_available() else "cpu")
net = VLM()
net.to(device)
ckpt_path = "~/MedFILIP/train/pretrained/model.pt"
checkpoints = torch.load(ckpt_path, map_location=device)
net.load_state_dict(checkpoints["network"])
net.eval()
dataset = np.load("~/MedFILIP/FT/mimic_retrieve.npy", allow_pickle=True)
text_datas = []
image_datas = []
for data in dataset:
    text_datas.append(data["labels"])
    image_datas.append(np.array(data["image_npy"]))
labels = [
    [text_datas[i] == text_datas[j] for j in range(len(text_datas))]
    for i in range(len(text_datas))
]
labels = np.array(labels)
text_embeddings = []
image_embeddings = []
b = 20
for i in range(int(len(text_datas) / b)):
    text_data = text_datas[i * b : (i + 1) * b]
    text_data = [sublist[0] for sublist in text_data]
    image_data = image_datas[i * b : (i + 1) * b]
    image_data = torch.tensor(image_data)
    image_data = image_data.to(device)
    image_embedding, text_embedding = net(image_data, text_data, device, True)
    text_embeddings.append(text_embedding.detach().cpu().numpy())
    image_embeddings.append(image_embedding.detach().cpu().numpy())
text_embeddings = np.concatenate(text_embeddings, axis=0)
image_embeddings = np.concatenate(image_embeddings, axis=0)
similarities = np.dot(image_embeddings, text_embeddings.T)
similarities = np.array(similarities)
dic_ItoT = {}

for k in range(1, 11):
    top_k_pred = np.argsort(similarities, axis=1)[:, -k:]
    top_k_pred = list(top_k_pred)
    label = [np.where(row == 1)[0] for row in labels]
    overlap_counts = [
        len(set(row1) & set(row2)) for row1, row2 in zip(top_k_pred, label)
    ]
    dic_ItoT[k] = sum(overlap_counts)
print(dic_ItoT)

dic_TtoI = {}
similarities, labels = similarities.T, labels.T
for k in range(1, 11):
    top_k_pred = np.argsort(similarities, axis=1)[:, -k:]
    top_k_pred = list(top_k_pred)
    label = [np.where(row == 1)[0] for row in labels]
    overlap_counts = [
        len(set(row1) & set(row2)) for row1, row2 in zip(top_k_pred, label)
    ]
    dic_TtoI[k] = sum(overlap_counts)
print(dic_TtoI)
