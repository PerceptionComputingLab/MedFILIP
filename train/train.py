from PIL import Image
import numpy as np
from tqdm import tqdm
import timeit
import os
import copy
import random
from datetime import datetime
import argparse
import json
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
import clip
from models import VLM
from constants import disease_type, description_book

IKI_dic, disease_type = description_book, disease_type[-16:]


def collate_fn(data):
    """
    Custom data collation function to process batches of data from the data loader.
    This function separates images, labels, and texts into individual lists for further processing.

    Parameters:
    - data: A list of tuples, each containing an image (NumPy array), a label (integer), and a text (string).

    Returns:
    - data_copy: A tuple containing three elements:
        1. images: A NumPy array of all images.
        2. labels: A list of all labels.
        3. texts: A list of all texts.
    """
    # Initialize lists for images, labels, and texts
    images, labels, texts = [], [], []

    # Iterate over the batch and append images, labels, and texts to their respective lists
    for batch in range(0, len(data)):
        images.append(data[batch][0])
        labels.append(data[batch][1])
        texts.append(data[batch][2])

    # Convert the list of images to a NumPy array and return it with labels and texts
    data_copy = (np.array(images), labels, texts)
    return data_copy


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def draw_similarity_matrix(similarity_matrix, name, size=(80, 60)):
    plt.figure(figsize=size)
    sns.heatmap(similarity_matrix)
    plt.savefig(name)
    plt.close()


def SSM(text, entity, lengths, tokenizer, model, device):
    """
    Calculate the similarity between entities in the text.

    Parameters:
    text (List[str]): List of segmented texts.
    entity (List[str]): List of entities.
    lengths (List[int]): List of lengths for each entity.
    tokenizer: Tokenizer object used for tokenization.
    model: Model object used to generate embeddings.
    device: Device information (GPU or CPU).

    Returns:
    torch.Tensor: Similarity matrix between entities.
    """
    # Create a boolean matrix indicating whether each character in the entity matches others
    bool_matrix = [[item2 == item for item2 in entity] for item in entity]

    # Tokenize the text
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    encoded_input = {key: tensor.to(device) for key, tensor in encoded_input.items()}

    # Compute embeddings using the model without gradient calculation
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform mean pooling to obtain sentence embeddings
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    # Compute and retrieve the similarity matrix of sentence embeddings
    similarity_matrix = torch.matmul(sentence_embeddings, sentence_embeddings.T)
    similarity_matrix = similarity_matrix.cpu().numpy()

    # Apply the boolean matrix to filter non-similar items
    similarity_matrix = bool_matrix * (similarity_matrix)

    # Insert 0 at the beginning of the lengths list and convert it to a cumulative sum array
    lengths.insert(0, 0)
    lengths = np.array(lengths)
    lengths = np.cumsum(lengths)

    # Split and compute the maximum value of the similarity matrix based on entity lengths
    similarity_matrix = [
        np.max(similarity_matrix[lengths[i] : lengths[i + 1]], axis=0)
        for i in range(len(lengths) - 1)
    ]

    # Return the processed similarity matrix
    return torch.tensor(np.array(similarity_matrix))


def load_png(img_path):
    img = Image.open(img_path).convert("L")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((448, 448), antialias=True),
        ]
    )
    return transform(img)


def data_prepare(save_path):
    """
    Prepare the dataset and save it in numpy format.

    This function reads data from a JSON file, processes the images, calculates the mean and standard deviation,
    normalizes the images, and saves the processed data to the specified path.

    Parameters:
    - save_path: str, path to save the processed data.
    """
    # Load data from JSON file
    with open("./data_llama3_8B.json", "r", encoding="utf-8") as file:
        datas = json.load(file)

    # Initialize image list
    image_list = []
    # Process only the first 1000 data entries
    for data in datas[:1000]:
        # Load and save the image
        image_npy = load_png(data["images"][0])
        image_list.append(image_npy)

    # Convert image list to numpy array
    image_list = np.array(image_list)

    # Calculate mean and standard deviation of images
    mean_value = np.mean(image_list, axis=(0, 2, 3))
    std_value = np.std(image_list, axis=(0, 2, 3))

    # Define image transformation for normalization
    transform = transforms.Compose(
        [transforms.Normalize(mean=mean_value, std=std_value)]
    )

    # Print mean and standard deviation
    print(f"Mean: {mean_value}")
    print(f"Standard Deviation: {std_value}")

    # Initialize array to save processed data
    data_tosave = np.array([])

    # Record start time
    time0 = timeit.default_timer()

    # Iterate over all data for processing
    for i in range(len(datas[:])):
        # Print progress every 1000 data entries
        if i % 1000 == 0:
            print("i", i)
        data = datas[i]

        # Process labels
        labels = []
        for s in data["response"].split(", "):
            s = s.split(" ")
            labels.append([s[0], " ".join(s[1:-1]), s[-1]])

        # Update data and remove unnecessary fields
        data["labels"] = labels
        del data["response"]
        del data["query"]

        # Load and transform the image
        img = load_png(data["images"][0])
        image_npy = transform(img)
        data["image_npy"] = image_npy

        # Append processed data to the save array
        data_tosave = np.append(data_tosave, data)

    # Save processed data to the specified path
    np.save(save_path, data_tosave)

    # Record end time and print elapsed time
    time1 = timeit.default_timer()
    print("Data loading: ", time1 - time0)


def random_mask(labels, p=0.5):
    for k in range(len(labels)):
        for i in range(0, len(labels[k]) - 1):
            if random.random() < p:
                labels[k][i] = "mask"
        if random.random() < p:
            labels[k][-1] = "mask"
    return labels


class Mimic_CXR(Dataset):
    def __init__(self, data, mode="train", mask="random_mask"):
        self.data = data
        if mode == "train":
            self.data = self.data[: int(len(self.data) * 0.9)]
        else:
            self.data = self.data[int(len(self.data) * 0.9) :]
        self.mask = mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = copy.deepcopy(self.data[idx]["labels"])
        # Inject Image Knowledge Injector module to inject image-specific knowledge into the model
        text = [
            (
                " ".join(sublist) + ". " + description_book[sublist[2]]
                if sublist[2] in description_book
                else " ".join(sublist)
            )
            for sublist in label
        ]
        return np.array(self.data[idx]["image_npy"].repeat(3, 1, 1)), label, text


def calculate_auc(fpr, tpr):
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    auc = np.trapz(tpr, fpr)
    return auc


def compute_metrics(predicted, labels):
    """
    Compute evaluation metrics for the model.

    Parameters:
    predicted: Model predictions, same shape as labels.
    labels: Ground truth labels, same shape as predicted.

    Returns:
    Strict accuracy, overall accuracy, precision, recall, F1 score, true positive rate, false positive rate, 
    true positives, false positives, true negatives, false negatives.
    """
    # Compute strict accuracy, i.e., proportion where all predictions are correct
    correct_predictions = np.all(predicted == labels)
    num_correct = np.sum(correct_predictions)
    num_total = predicted.shape[0]
    accuracy_strict = num_correct / num_total

    # Compute components of the confusion matrix
    TP = np.sum((predicted == 1) & (labels == 1))
    FP = np.sum((predicted == 1) & (labels == 0))
    TN = np.sum((predicted == 0) & (labels == 0))
    FN = np.sum((predicted == 0) & (labels == 1))

    # Compute various evaluation metrics based on the confusion matrix
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    # Return all computed evaluation metrics and confusion matrix elements
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


def filter_array(arr):
    return arr[(arr >= 0) & (arr <= 1)]


def eval(net, dataloader, device):
    """
    Evaluate model performance.

    Parameters:
    - net: Model network.
    - dataloader: Data loader for iterating over data.
    - device: Device information (CPU or GPU).

    Returns:
    - f1_list: List of F1 scores.
    - auc_list: List of AUC scores.
    - acc_list: List of accuracies.
    """
    with torch.set_grad_enabled(False):
        # Initialize concatenated lists for predictions and labels
        predicted_concat = []
        label_concat = []

        # Iterate over data in the data loader
        for images, labels, texts in tqdm(dataloader):
            # Convert image data to tensor and move to specified device
            images = torch.tensor(images)
            images = images.to(device)

            # Extract entity information from labels
            entity = [[arr[2] for arr in label] for label in labels]
            lengths = [len(sublist) for sublist in entity]
            entity = [item for sublist in entity for item in sublist]

            # Generate ground truth labels
            ground_truth = [
                [item2 == item for item2 in disease_type] for item in entity
            ]
            ground_truth = np.array(ground_truth)
            ground_truth = ground_truth * np.ones(ground_truth.shape)

            # Accumulate lengths for subsequent processing
            lengths.insert(0, 0)
            lengths = np.array(lengths)
            lengths = np.cumsum(lengths)

            # Perform max aggregation on ground truth
            ground_truth = [
                np.max(ground_truth[lengths[i] : lengths[i + 1]], axis=0)
                for i in range(len(lengths) - 1)
            ]
            labels_np = np.array(ground_truth)

            # Generate prompt text
            prompt_text = [
                disease + ". " + description_book[disease] for disease in disease_type
            ]

            # Use the model to make predictions
            predicted_np = net(images, prompt_text, device).detach().cpu().numpy()

            # Append predictions and labels to concatenated lists
            predicted_concat.append(predicted_np)
            label_concat.append(labels_np)

        # Concatenate predictions and labels along the specified axis
        predicted_concat = np.concatenate(predicted_concat, axis=0)
        label_concat = np.concatenate(label_concat, axis=0)

        # Initialize results list
        results = []
        # Define threshold range
        thresholds = np.arange(-1, 1, 0.001)

        # Iterate over each threshold
        for threshold in thresholds:
            # Copy predictions and apply threshold
            predicted_binary = copy.deepcopy(predicted_concat)
            predicted_binary = np.where(predicted_binary >= threshold, 1, 0)

            # Initialize performance metric result list
            result = []
            for c in range(predicted_binary.shape[1]):
                # Compute performance metrics for each class
                metric = compute_metrics(predicted_binary[:, c], label_concat[:, c])
                result.append(metric)
            results.append(result)

        # Initialize performance metric lists
        auc_list = []
        f1_list = []
        acc_list = []

        # Convert results to numpy array
        results = np.array(results)

        # Iterate over each class
        for c in range(predicted_binary.shape[1]):
            result = results[:, c, :]

            # Filter F1 scores
            f1_ = filter_array(result[:, 4])
            if len(f1_) == 0:
                continue

            # Add maximum F1 score to the list
            f1_list.append(np.max(f1_))

            # Add corresponding accuracy for maximum F1 score to the list
            acc_ = result[:, 1][np.where(result[:, 4] == np.max(f1_))]
            acc_list.append(np.max(acc_))

            # Compute and add AUC score to the list
            auc_list.append(calculate_auc(result[:, 6][::-1], result[:, 5][::-1]))

        # Return performance metric lists
        return f1_list, auc_list, acc_list


def parse_args():
    parser = argparse.ArgumentParser(description="Train a classifier")
    parser.add_argument("--tag", type=str, help="Experiment tag", default="experiment")
    parser.add_argument("--gpu", type=int, help="GPU index", default=0)
    parser.add_argument("--vision_model", type=str, help="Vision encoder", default="resnet")
    parser.add_argument("--log_path", type=str, help="Training log path", default="./log")
    parser.add_argument(
        "--checkpoints_path", type=str, help="Model checkpoint path", default="./checkpoints"
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        help="Pretrained model path",
        default="",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Set PyTorch CUDA memory allocation configuration to limit max split size to 128MB
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # If torch.cuda has an empty_cache method, call it to clear the cache
    if hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()

    # Record start time for data loading
    time0 = timeit.default_timer()

    # Initialize a dictionary to record time consumption for different stages
    time = {"Data Loading": 0, "Label Generation": 0, "Model Training": 0}

    # Set CUDA launch blocking mode for debugging
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Disable parallelism in tokenizers to avoid potential conflicts
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Parse command-line arguments
    args = parse_args()

    # Get the tag information from command-line arguments
    tag = str(args.tag)

    # Set the maximum number of epochs for training
    epoches = 20

    # Select device (CPU or GPU) based on availability
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    )

    # Get log path and checkpoint path from command-line arguments
    log_path = args.log_path
    checkpoints_path = args.checkpoints_path

    # Get current timestamp to generate unique folder names
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")

    # Construct log and checkpoint paths with timestamps
    log_path = os.path.join(log_path, "{}_".format(tag) + current_time)
    checkpoints_path = os.path.join(checkpoints_path, "{}_".format(tag) + current_time)

    # Create directories for logs and checkpoints
    os.makedirs(log_path)
    os.makedirs(checkpoints_path)

    # Initialize TensorBoard SummaryWriter for logging training progress
    writer = SummaryWriter(log_dir=log_path)

    # Define training data path and call data_prepare function for data preprocessing
    data_path = "./mimic_train.npy"
    data_prepare(data_path)

    # Load preprocessed data
    datas = np.load(data_path, allow_pickle=True)

    # Create training dataset and data loader
    train_dataset = Mimic_CXR(datas)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Create evaluation dataset and data loader
    eval_dataset = Mimic_CXR(
        datas,
        mode="eval",
        mask="all_mask",
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Load pretrained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("/home/lxj/.cache/all-mpnet-base-v2/")
    similarity_model = AutoModel.from_pretrained("/home/lxj/.cache/all-mpnet-base-v2/")

    # Move similarity model to the specified device (CPU or GPU)
    similarity_model.to(device)

    # Initialize the custom VLM model
    net = VLM(vision_model=args.vision_model)

    # Move VLM model to the specified device (CPU or GPU)
    net.to(device)

    # If a pretrained model path is provided, load the pretrained model weights
    if args.pretrained_path:
        checkpoints = torch.load(args.pretrained_path, map_location=device)
        net.load_state_dict(checkpoints["network"])

    # Define loss functions: Mean Squared Error Loss, Cross Entropy Loss, and Binary Cross Entropy Loss
    MSE_Loss = nn.MSELoss()
    CE_Loss = nn.CrossEntropyLoss()
    BCE_Loss = nn.BCEWithLogitsLoss()

    # Define optimizer: Stochastic Gradient Descent (SGD) with learning rate 0.0001 and weight decay 1e-4
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, weight_decay=1e-4)

    # Define learning rate scheduler: Cosine Annealing Warm Restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=2 * len(train_dataloader),
        T_mult=1,
        eta_min=1e-6,
        last_epoch=-1,
    )

    # Record the end time of data loading and print the time taken for data loading
    time1 = timeit.default_timer()
    print("Data loading: ", time1 - time0)
    time["Data Loading"] += time1 - time0

    # Reinitialize the timer to record time for other stages
    time1 = timeit.default_timer()

    # Initialize the best score to 0
    best_score = 0

    # Start the training loop, iterating for the specified number of epochs
    for epoch in range(epoches):
        # Initialize loss list for each epoch
        loss_epoch = [0, 0, 0, 0]

        # Iterate over each batch of data in the training data loader
        for images, labels, texts in tqdm(train_dataloader):
            # Record the start time of label generation
            time1 = timeit.default_timer()

            # Convert image data to tensor and move to the specified device (CPU or GPU)
            images = torch.tensor(images)
            images = images.to(device)

            # Flatten texts into a single list of strings
            texts = [sequence for sublist in texts for sequence in sublist]

            # Extract entity information from labels
            entity = [[arr[2] for arr in label] for label in labels]

            # Calculate the length of each entity in the samples
            lengths = [len(sublist) for sublist in entity]

            # Flatten the entity list
            entity = [item for sublist in entity for item in sublist]

            # Generate semantic similarity label matrix
            mse_label = SSM(
                copy.deepcopy(texts),
                copy.deepcopy(entity),
                copy.deepcopy(lengths),
                tokenizer,
                similarity_model,
                device,
            )

            # Move the semantic similarity label matrix to the specified device (CPU or GPU)
            mse_label = mse_label.to(device)

            # Insert 0 at the beginning of the lengths list and compute the cumulative sum
            lengths.insert(0, 0)
            lengths = np.array(lengths)
            lengths = np.cumsum(lengths)

            # Record the end time of label generation and update label generation time
            time2 = timeit.default_timer()
            time["Label Generation"] += time2 - time1

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass to get model output
            logits_per = net(images, texts, device)

            # Compute MSE loss
            mse_loss = MSE_Loss(logits_per, mse_label)

            # Compute CE loss
            ce_loss = CE_Loss(logits_per, mse_label) / logits_per.size()[0]

            # Compute BCE loss
            bce_loss = BCE_Loss(logits_per, mse_label)

            # Compute total loss
            total_loss = mse_loss + bce_loss

            # Backpropagation and update model parameters
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # Record the end time of model training and update model training time
            time3 = timeit.default_timer()
            time["Model Training"] += time3 - time2

            # Accumulate losses for each batch
            loss_epoch[0] += total_loss.item()
            loss_epoch[1] += mse_loss.item()
            loss_epoch[2] += ce_loss.item()
            loss_epoch[3] += bce_loss.item()

        # Evaluate model performance at the end of each epoch
        scores = eval(net, eval_dataloader, device)
        f1_score = np.mean(scores[0])
        auc_score = np.mean(scores[1])
        acc_score = np.mean(scores[2])

        # Print information for the current epoch
        len_ = len(train_dataloader)
        print("epoch: ", epoch)
        print(time)
        print("epoch_loss: ", loss_epoch[0] / len_)
        print(scores[0])
        print("f1_score: ", f1_score)
        print(scores[1])
        print("auc_score: ", auc_score)
        print(scores[2])
        print("acc_score: ", acc_score)

        # Write evaluation results to the log file
        with open(os.path.join(log_path, "logs.txt"), "a") as logs:
            if epoch % 20 == 0:
                logs.write(
                    "{:<10}|{:<8}|{:<9}|{:<9}|{:<9}|{:<9}|{:<9}|{:<9}|{:<9}\n".format(
                        " lr",
                        " epoch",
                        " total",
                        " mse",
                        " ce",
                        " bce",
                        " f1",
                        " auc",
                        " acc",
                    )
                )
            logs.write(
                str(
                    " %0.6f | %6d | %0.5f | %0.5f | %0.5f | %0.5f | %0.5f | %0.5f | %0.5f "
                    % (
                        optimizer.state_dict()["param_groups"][0]["lr"],
                        epoch,
                        loss_epoch[0] / len_,
                        loss_epoch[1] / len_,
                        loss_epoch[2] / len_,
                        loss_epoch[3] / len_,
                        f1_score,
                        auc_score,
                        acc_score,
                    )
                )
                + "\n"
            )

        # Use TensorBoard to log evaluation results
        writer.add_scalars(
            "time",
            {
                "epoch_loss": loss_epoch[0] / len_,
                "mse_loss": loss_epoch[1] / len_,
                "ce_loss": loss_epoch[2] / len_,
                "bce_loss": loss_epoch[3] / len_,
                "f1_score": f1_score,
                "auc_score": auc_score,
                "acc_score": acc_score,
            },
            epoch,
        )

        # Save the model if the current F1 score is better than the best score
        if f1_score > best_score:
            best_score = f1_score
            torch.save(
                {
                    "it": epoch,
                    "network": net.state_dict(),
                    "image_model": net.image_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                os.path.join(checkpoints_path, f"score_{best_score:.3f}.pt"),
            )

        # Save the model every 5 epochs
        if epoch % 5 == 0:
            torch.save(
                {
                    "it": epoch,
                    "network": net.state_dict(),
                    "image_model": net.image_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                os.path.join(
                    checkpoints_path, f"epoch_{epoch}_score_{f1_score:.3f}.pt"
                ),
            )