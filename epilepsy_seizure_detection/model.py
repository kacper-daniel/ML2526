import torch
from torch import device, nn
from tqdm import tqdm
import numpy as np

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

def train(model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        n_epochs: int,
        batch_size: int,
        device):
    
    train_loss_log = []
    train_acc_log = []
    val_loss_log = []
    val_acc_log = []

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = train_loss / (len(train_dataloader) / batch_size)
        epoch_acc = (correct / total)*100

        model.eval()
        val_loss = 0.0
        val_total = 0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / (len(val_dataloader) / batch_size)
        val_acc = (val_correct/val_total)*100
        print(f'[Epoch: {epoch + 1}] Train loss: {epoch_loss:.2f} | Train acc: {epoch_acc:.2f}% | Val loss: {val_loss:.2f} | Val acc: {val_acc:.2f}%')
        train_loss_log.append(epoch_loss)
        train_acc_log.append(epoch_acc)
        val_acc_log.append(val_acc)
        val_loss_log.append(val_loss)

    fig1, ax_acc = plt.subplots()
    plt.plot(train_acc_log)
    plt.plot(val_acc_log)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model - Accuracy')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()
    
    fig2, ax_loss = plt.subplots()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model - Loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.plot(train_loss_log)
    plt.plot(val_loss_log)
    plt.show()

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
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss = test_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)

    print(f"\n--- Test Results ---")
    print(f"Loss:      {avg_loss:.4f}")
    print(f"Accuracy:  {acc*100:.2f}%")