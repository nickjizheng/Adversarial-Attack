import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import random  # Import random for shuffling data
import shutil  # Import shutil for file operations
import cv2
from PIL import Image  # Import Image from PIL for image processing
import matplotlib.pyplot as plt  # Import for plotting charts
from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DATA_DIR = Path(os.getenv("UCF101_DATA_DIR", PROJECT_ROOT / "data" / "UCF101"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "outputs"))
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints"))
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)


# Defining the HumanActivityDataset within the same file
class HumanActivityDataset(Dataset):
    def __init__(self, data_dir, class_to_idx, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.data = []
        for class_dir in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, class_dir)):
                for file_name in os.listdir(os.path.join(data_dir, class_dir)):
                    if file_name.endswith(('.mp4', '.avi')):
                        self.data.append((os.path.join(data_dir, class_dir, file_name), class_dir))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, class_dir = self.data[idx]
        label = self.class_to_idx[class_dir]

        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        # Convert frames to PIL images
        if self.transform:
            frames = [self.transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))) for frame in frames]

        frames = torch.stack(frames, dim=0)

        return frames, label

# Your training and evaluation script starts here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_class_to_idx(data_dir):
    classes = sorted(entry.name for entry in os.scandir(data_dir) if entry.is_dir())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    return class_to_idx

import torch.cuda.amp as amp

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=10):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    scaler = amp.GradScaler()  # Mixed precision training scaler
    
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            start_time = time.time()

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                batch_start_time = time.time()
                batch_size, num_frames, channels, height, width = inputs.size()
                inputs = inputs.view(batch_size * num_frames, channels, height, width).to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with amp.autocast():  # Mixed precision context
                        outputs = model(inputs)
                        outputs = outputs.view(batch_size, num_frames, -1).mean(dim=1)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            end_time = time.time()
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {end_time - start_time:.2f}s')

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
                scheduler.step()
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    
    # Plotting the loss and accuracy curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_loss_history, label='Training Loss')
    plt.plot(range(num_epochs), val_loss_history, label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), train_acc_history, label='Training Accuracy')
    plt.plot(range(num_epochs), val_acc_history, label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save the figure
    plt.savefig(OUTPUT_DIR / 'training_progress.png')
    plt.show()

    return model

def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            batch_size, num_frames, channels, height, width = inputs.size()
            inputs = inputs.view(batch_size * num_frames, channels, height, width).to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            outputs = outputs.view(batch_size, num_frames, -1).mean(dim=1)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels.data)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    total_loss = running_loss / (len(dataloader.dataset))
    total_acc = running_corrects.double() / len(dataloader.dataset)
    
    print(f'Test Loss: {total_loss:.4f} Acc: {total_acc:.4f}')
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.png')
    plt.show()
    
    return total_loss, total_acc

def split_data(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum up to 1")

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        files = [f for f in os.listdir(cls_dir) if f.endswith(('.mp4', '.avi'))]
        random.shuffle(files)
        
        train_cutoff = int(train_ratio * len(files))
        val_cutoff = int((train_ratio + val_ratio) * len(files))
        
        split_files = {
            'train': files[:train_cutoff],
            'val': files[train_cutoff:val_cutoff],
            'test': files[val_cutoff:]
        }
        
        for split in split_files:
            split_cls_dir = os.path.join(data_dir, split, cls)
            os.makedirs(split_cls_dir, exist_ok=True)
            for file_name in split_files[split]:
                src_file = os.path.join(cls_dir, file_name)
                dst_file = os.path.join(split_cls_dir, file_name)
                shutil.copy(src_file, dst_file)

def main():
    # Path to the dataset
    data_dir = str(TRAINING_DATA_DIR)

    # Split the data into train, val, and test sets
    split_data(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    class_to_idx = get_class_to_idx(os.path.join(data_dir, 'train'))
    class_labels = list(class_to_idx.keys())
    
    print("Class to Index Mapping:", class_to_idx)
    print("Class Labels:", class_labels)
    assert len(class_labels) == len(class_to_idx), "Mismatch between class labels and class to index mapping."

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        x: HumanActivityDataset(os.path.join(data_dir, x), class_to_idx, transform=data_transforms[x])
        for x in ['train', 'val', 'test']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
        for x in ['train', 'val', 'test']
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    image_model = models.resnet18(pretrained=True)
    num_features = image_model.fc.in_features
    image_model.fc = nn.Linear(num_features, len(class_labels))
    image_model = image_model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(image_model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    num_epochs = 10
    
    best_model = train_model(image_model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=num_epochs)
    
    # Save the fine-tuned model
    torch.save(best_model.state_dict(), CHECKPOINT_DIR / 'fine_tuned_resnet18_final.pth')
    print('Model saved successfully.')
    
    # Evaluate the model on the test set
    test_loss, test_acc = evaluate_model(best_model, dataloaders['test'], criterion)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

if __name__ == "__main__":
    main()
