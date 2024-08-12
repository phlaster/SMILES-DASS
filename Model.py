import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import tifffile
from utils import *
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self, layers):
        super(CNN, self).__init__()
        self.train_story = []
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

    def calculate_weights(self, class_counts):
        total_pixels = class_counts.sum()
        class_frequencies = class_counts.float() / total_pixels
        median_frequency = torch.median(class_frequencies)
        weights = median_frequency / class_frequencies
        return weights

    def train(self,
        train_loader, val_loader,
        num_epochs,
        learning_rate,
        thresholdmetric=0.6,
        saving_name="model.torch",
        gamma=2.0,
        alpha=0.5
    ):
        device = get_device()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
        self.model.to(device)

        for epoch in range(num_epochs):
            train_loss = self._train_epoch(epoch, num_epochs, train_loader.loader, optimizer, device, gamma, alpha)
            scores = self._validate_epoch(epoch, num_epochs, val_loader.loader, device, thresholdmetric, saving_name, gamma, alpha)

            current_lr = optimizer.param_groups[0]['lr']

            self.train_story.append({
                "learn_rate": current_lr,
                "train_loss": train_loss,
                "val_loss": scores[0],
                "val_accuracy": scores[1],
                "val_recall": scores[2],
                "val_precision": scores[3],
                "val_f1": scores[4]
            })
            scheduler.step()
    
    def _train_epoch(self, epoch, num_epochs, train_loader, optimizer, device, gamma, alpha):
        self.model.train()
        running_loss = 0.0
        for images, masks, class_counts in tqdm(train_loader, desc=f'Train {epoch+1}/{num_epochs}', leave=False):
            images, masks = images.to(device), masks.to(device)
            class_counts = class_counts.to(device)

            batch_weights = self.calculate_weights(class_counts.sum(dim=0))
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = focal_loss(outputs, masks, gamma, alpha, batch_weights)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = round(running_loss/len(train_loader), 2)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train loss: {epoch_loss}', end="")
        return epoch_loss

    def _validate_epoch(self, epoch, num_epochs, val_loader, device, thresholdmetric, saving_name, gamma, alpha):
        self.model.eval()
        running_loss = 0.0
        num_classes = 5
        L = len(val_loader)
        accuracy_scores = np.zeros(num_classes)
        recall_scores = np.zeros(num_classes)
        precision_scores = np.zeros(num_classes)
        f1_scores = np.zeros(num_classes)
        
        with torch.no_grad():
            for images, masks, class_counts in tqdm(val_loader, desc=f'Validate {epoch+1}/{num_epochs}', leave=False):
                images, masks = images.to(device), masks.to(device)
                class_counts = class_counts.to(device)
                batch_weights = self.calculate_weights(class_counts.sum(dim=0))                
                outputs = self.model(images)
                loss = focal_loss(outputs, masks, gamma, alpha, batch_weights)
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                predicted_np = predicted.cpu().numpy()
                masks_np = masks.cpu().numpy()
                accuracy_scores += metric_accuracy(masks_np, predicted_np, num_classes)
                recall_scores += metric_recall(masks_np, predicted_np, num_classes)
                precision_scores += metric_precision(masks_np, predicted_np, num_classes)
                f1_scores += metric_f1(masks_np, predicted_np, num_classes)

        val_loss = round(running_loss / L, 2)
        accuracy = round(np.mean(accuracy_scores/L), 2)
        recall = round(np.mean(recall_scores/L), 2)
        precision = round(np.mean(precision_scores/L), 2)
        f1 = round(np.mean(f1_scores/L), 2)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Val loss {val_loss}, Recall {recall}, precision {precision}, F1 {f1}', end="")
        meanmetric = round(np.mean([recall,precision]), 3)
        
        if meanmetric > thresholdmetric:
            self.pickle(saving_name)
            thresholdmetric = meanmetric + 0.01
        return val_loss, accuracy, recall, precision, f1
        

        
    def test(self, test_loader, gamma=2.0, alpha=0.5):
        device = get_device()
        self.model.to(device)
        self.model.eval()
        running_loss = 0.0
        num_classes = 5
        L = len(test_loader.loader)
        accuracy_scores = np.zeros(num_classes)
        recall_scores = np.zeros(num_classes)
        precision_scores = np.zeros(num_classes)
        f1_scores = np.zeros(num_classes)
        with torch.no_grad():
            for images, masks, class_counts in tqdm(test_loader.loader, desc='Testing...', leave=False):
                images, masks = images.to(device), masks.to(device)
                class_counts = class_counts.to(device)
                batch_weights = self.calculate_weights(class_counts.sum(dim=0))
                outputs = self.model(images)
                loss = focal_loss(outputs, masks, gamma, alpha, batch_weights)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                predicted_np = predicted.cpu().numpy()
                masks_np = masks.cpu().numpy()
                
                accuracy_scores += metric_accuracy(masks_np, predicted_np, num_classes)
                recall_scores += metric_recall(masks_np, predicted_np, num_classes)
                precision_scores += metric_precision(masks_np, predicted_np, num_classes)
                f1_scores += metric_f1(masks_np, predicted_np, num_classes)
        return {
            "test_loss":round(running_loss / L, 3),
            "test_accuracy":round(np.mean(accuracy_scores / L), 3),
            "test_recall":round(np.mean(recall_scores / L), 3),
            "test_precision":round(np.mean(precision_scores / L), 3),
            "test_f1":round(np.mean(f1_scores / L), 3)
        }
 
    def _preprocess_image(self, image_path, target_size=512):
        image = tifffile.imread(image_path)
        image = np.moveaxis(image, [0, 1, 2], [1, 2, 0])
        height, width = image.shape[1:3]
        pad_height = max(0, target_size - height)
        pad_width = max(0, target_size - width)
        padded_image = np.pad(image, ((0, 0), (0, pad_height), (0, pad_width)), mode='reflect')
        return padded_image.astype(np.float32)

    
    def predict(self, image_path):
        device = get_device()
        image = self._preprocess_image(image_path)
        image = torch.from_numpy(image).unsqueeze(0).to(device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
            predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        return predicted_mask
    
    def pickle(self, path):
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'train_story': self.train_story
        }
        torch.save(save_dict, path)
        print("Model and train_story saved")
        
    def unpickle(self, path):
        save_dict = torch.load(path)
        self.model.load_state_dict(save_dict['model_state_dict'])
        self.train_story = save_dict['train_story']
        print("Model and train_story loaded")

    def plot_training_history(self):
        epochs = list(range(1, len(self.train_story) + 1))
        train_loss = [entry['train_loss'] for entry in self.train_story]
        val_loss = [entry['val_loss'] for entry in self.train_story]
        # val_accuracy = [np.mean(entry['val_accuracy']) for entry in self.train_story]
        val_recall = [np.mean(entry['val_recall']) for entry in self.train_story]
        val_precision = [np.mean(entry['val_precision']) for entry in self.train_story]
        val_f1 = [np.mean(entry['val_f1']) for entry in self.train_story]
        learning_rate = [np.mean(entry['learn_rate']) for entry in self.train_story]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

        # 1
        # ax1.plot(epochs, val_accuracy, label='Validation Accuracy', color='tab:green', linestyle='-', marker='o')
        ax1.plot(epochs, val_recall, label='Validation Recall', color='tab:red', linestyle='-', marker='o')
        ax1.plot(epochs, val_precision, label='Validation Precision', color='tab:purple', linestyle='-', marker='o')
        ax1.plot(epochs, val_f1, label='Validation F1', color='tab:brown', linestyle='-', marker='o')

        ax1.set_ylabel('Metrics')
        ax1.set_title('Validation Metrics Over Epochs')
        ax1.grid(True)
        ax1.legend(loc='upper left')

        # 2
        ax2.plot(epochs, train_loss, label='Train Loss', color='tab:blue', linestyle='-', marker='o')
        ax2.plot(epochs, val_loss, label='Validation Loss', color='tab:orange', linestyle='-', marker='o')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training and Validation Loss Over Epochs')
        ax2.grid(True)
        ax2.legend(loc='upper left')

        # 3
        ax2_twin = ax2.twinx()
        ax2_twin.plot(epochs, learning_rate, label='Learning Rate', color='tab:gray', linestyle='--', marker='x')
        ax2_twin.set_ylabel('Learning Rate')
        ax2_twin.legend(loc='upper right')

        plt.tight_layout()
        plt.show()
        
def load_model(path, layers):
    model = CNN(layers)
    checkpoint = torch.load(path)
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.train_story = checkpoint['train_story']
    print("Model and train_story loaded")
    return model



def focal_loss(outputs, targets, gamma=2.0, alpha=0.5, weights=None):
    ce_loss = nn.CrossEntropyLoss(weight=weights)(outputs, targets)
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss
