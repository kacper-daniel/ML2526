import torch
from torch import device, nn
from tqdm import tqdm
import numpy as np

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

def train(model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        n_epochs: int,
        batch_size: int,
        device):
    
    loss_log = []
    acc_log = []

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = train_loss / (len(dataloader) / batch_size)
        epoch_acc = (correct / total)*100
        print(f'[Epoch: {epoch + 1}] Train loss: {epoch_loss:.2f} | Train acc: {epoch_acc:.2f}%')
        loss_log.append(epoch_loss)
        acc_log.append(epoch_acc)

    # plt.plot(range(n_epochs), loss_log, label='Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss Curve')
    # plt.legend()
    # plt.show()

    # plt.plot(range(n_epochs), acc_log, label='Training Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Training Accuracy Curve')
    # plt.legend()
    # plt.show()

def test(model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn,
        device):

    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    avg_loss = test_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"\n--- Test Results ---")
    print(f"Loss:      {avg_loss:.4f}")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    class_names = dataloader.dataset.classes
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(
        all_labels, 
        all_preds, 
        display_labels=class_names,
        cmap='Blues', 
        ax=ax,
        colorbar=False,
        xticks_rotation=45
    )
    plt.title('Confusion Matrix')
    plt.show()
