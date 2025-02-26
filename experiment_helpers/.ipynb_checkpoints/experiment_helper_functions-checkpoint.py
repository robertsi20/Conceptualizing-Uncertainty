import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset

import urllib.request
from fastai.vision.all import *
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import glob
from torchvision import transforms
import random



# Add dropout to both g and h
class StochasticModel(nn.Module):
    def __init__(self, h, dropout_prob=0.5):
        super(StochasticModel, self).__init__()
        # self.g = nn.Sequential(
        #     g,
        #     nn.Dropout(p=dropout_prob)  # Add dropout after g
        # )
        self.h = nn.Sequential(
            nn.Dropout(p=dropout_prob),  # Add dropout before logits
            h
        )

    def forward(self, x):
        # x = self.g(x)
        self.h.train()
        x = self.h(x)
        return x

# Function to filter dataset by random classes
def filter_dataset_by_classes(dataset, num_classes):
    classes = random.sample(range(len(dataset.classes)), num_classes)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    subset = Subset(dataset, indices)
    subset.classes = [dataset.classes[c] for c in classes]
    subset.class_to_idx = class_to_idx
    return subset

def predict_with_uncertainty_batched(f_model, data_loader, n_iter=10, device="cpu"):
    """
    Perform N stochastic forward passes over batches of data and return predictions.

    Args:
        f_model (nn.Module): The model with dropout layers.
        data_loader (DataLoader): DataLoader for the dataset.
        n_iter (int): Number of stochastic forward passes.
        device (str): Device to perform computations on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Predictions of shape (n_iter, num_data, num_classes).
    """
    # f_model.train()  # Ensure dropout is active during inference

    # Store predictions for all iterations
    all_preds = []
    for _ in range(n_iter):
        preds = []
        for inputs, _ in data_loader:
            with torch.no_grad():
                batch_preds = torch.softmax(f_model(inputs), dim=1)

            preds.append(batch_preds)
        # Concatenate predictions for this iteration
        all_preds.append(torch.cat(preds, dim=0))

    # Stack predictions across iterations
    return torch.stack(all_preds, dim=0)

# Function to preprocess the images in batches to avoid OOM errors
def preprocess_images_in_batches(filelist,func, batch_size=64):
    all_images, all_labels = [], []
    for start_idx in range(0, len(filelist), batch_size):
        end_idx = min(start_idx + batch_size, len(filelist))
        batch_files = filelist[start_idx:end_idx]
        x, y = zip(*func(batch_files))
        x, y = np.array(x), np.array(y)
        images_preprocessed = torch.stack([transform(to_pil(img)) for img in x], 0)
        all_images.append(images_preprocessed)
        all_labels.append(torch.from_numpy(y))
    return torch.cat(all_images), torch.cat(all_labels)
def load_imagenette():
    device = 'cuda'

    # loading any timm model
    model = timm.create_model('nf_resnet50.ra2_in1k', pretrained=True)
    model = model.to(device)

    # processing
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    to_pil = transforms.ToPILImage()

    # cut the model in two parts
    g = nn.Sequential(*(list(model.children())[:4]))
    def gen_images(filelist):
        for f in filelist:
            folder_name = f.split('/')[-2]
            class_id = folder_name2class_id[folder_name]
            im = Image.open(f)
            if len(im.getbands()) == 3:
                yield np.array(im.resize((224, 224))), class_id

    with urllib.request.urlopen('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt') as f:
        imagenet_class_names = np.array(f.read().decode('utf-8').split('\n'))

    imagenette_folder = untar_data(URLs.IMAGENETTE_160)

    folders_names = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079', 'n03394916', 'n03417042',
                     'n03425413', 'n03445777', 'n03888257']
    imagenette_class_names = ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn',
                              'garbage truck', 'gas pump', 'golf ball', 'parachute']

    imagenette_class_ids = [np.where(imagenet_class_names == class_name)[0][0] for class_name in imagenette_class_names]
    folder_name2class_id = dict(zip(folders_names, imagenette_class_ids))
    # train_filelist = glob.glob(f'{imagenette_folder}/train/*/*.JPEG')
    val_filelist = glob.glob(f'{imagenette_folder}/val/*/*.JPEG')
    val_images, val_labels = preprocess_images_in_batches(val_filelist,gen_images)

    return val_images, val_labels

# Define a dataset class for embeddings
class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# Function to sample datasets
def sample_datasets(iid_dataset, ood_dataset, iid_count, ood_percentage):
    # Sample IID dataset
    iid_indices = random.sample(range(len(iid_dataset)), iid_count)
    iid_subset = Subset(iid_dataset, iid_indices)

    # Sample OOD dataset
    ood_count = int(len(ood_dataset) * ood_percentage)
    ood_indices = random.sample(range(len(ood_dataset)), ood_count)
    ood_subset = Subset(ood_dataset, ood_indices)

    # Merge datasets and create labels
    combined_dataset = ConcatDataset([iid_subset, ood_subset])
    iid_labels = torch.zeros(len(iid_subset))
    ood_labels = torch.ones(len(ood_subset))
    labels = torch.cat([iid_labels, ood_labels]).long()

    return combined_dataset, labels
def load_imagewoof():
    device = 'cuda'

    # loading any timm model
    model = timm.create_model('nf_resnet50.ra2_in1k', pretrained=True)
    model = model.to(device)

    # processing
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    to_pil = transforms.ToPILImage()

    # cut the model in two parts
    g = nn.Sequential(*(list(model.children())[:4]))
    def gen_images(filelist):
        for f in filelist:
            folder_name = f.split('/')[-2]
            class_id = folder_name2class_id[folder_name]
            im = Image.open(f)
            if len(im.getbands()) == 3:
                yield np.array(im.resize((224, 224))), class_id

    with urllib.request.urlopen('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt') as f:
        imagenet_class_names = np.array(f.read().decode('utf-8').split('\n'))

        # Load imagewoof dataset
    imagewoof_folder = untar_data(URLs.IMAGEWOOF_160)

    folders_names = ['n02086240', 'n02087394', 'n02088364', 'n02089973', 'n02093754', 'n02096294', 'n02099601',
                         'n02105641', 'n02111889', 'n02115641']

    imagewoof_class_ids = list(range(0, 10))
    folder_name2class_id = dict(zip(folders_names, imagewoof_class_ids))

        # Get file lists for train, val, and test sets
    val_filelist = glob.glob(f'{imagewoof_folder}/val/*/*.JPEG')
    val_images, val_labels = preprocess_images_in_batches(val_filelist)

    return val_images, val_labels




class ClassifierHead(nn.Module):
    def __init__(self, in_features=2048, out_features=10):
        super(ClassifierHead, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features, out_features)
        self.flatten = nn.Identity()

    def forward(self, x):
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)
        return x

def process_in_batches(data, batch_size, func):
    """
    Process data in batches using the full_wrapper function.

    Args:
        data (np.ndarray): Input data to process.
        batch_size (int): Size of each batch.

    Returns:
        np.ndarray: Predictions for the entire dataset.
    """
    # Initialize an empty list to store predictions
    all_preds = []

    # Iterate over the data in batches
    for start_idx in range(0, len(data), batch_size):
        # Get the end index for the current batch
        end_idx = min(start_idx + batch_size, len(data))
        # Get the current batch
        batch = data[start_idx:end_idx]
        # Use full_wrapper to process the batch and get predictions
        batch_preds = func(batch)
        # Append the predictions to the list
        all_preds.append(batch_preds)

    # Concatenate all batch predictions into a single numpy array
    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Train the neural network
def train_nn(train_loader, model, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            # print(outputs)
            # print(labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

# Evaluate the model
def evaluate_nn(test_loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(torch.softmax(outputs,dim=1), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout2(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x
