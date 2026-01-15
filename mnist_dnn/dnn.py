import numpy as np
import matplotlib.pyplot as plt
from layer import DenseLayer

class DenseNeuralNetwork():
    def __init__(self, input_size: int, output_size: int, hidden_size: tuple[int]):
        '''
            Neural network with 3 dense layers

            input_size (int): input vector length
            output_size (int): output vector length
            hidden_size (tuple(int)): hidden layer size
        '''
        self.layer1 = DenseLayer(input_size=input_size, output_size=hidden_size[0], activation="relu")
        self.layer2 = DenseLayer(input_size=hidden_size[0], output_size=hidden_size[1], activation="relu")
        self.layer3 = DenseLayer(input_size=hidden_size[1], output_size=output_size, activation="softmax")

        self.loss_log = []
        self.train_acc_log = []
        self.val_acc_log = []

    def forward(self, inputs):
        '''
            Forward pass

            inputs (Tensor): Data
        '''
        output1 = self.layer1.forward(inputs)
        output2 = self.layer2.forward(output1)
        output3 = self.layer3.forward(output2)

        return output3
    
    def fit(self, X_train, y_train, X_val, y_val, n_epochs: int, init_lr: float, decay: float, batch_size: int = 128):
        '''
            Training process
                Forward pass, loss computation, backpropagation

            X_train (Tensor): Data
            y_train (Tensor): Target labels
            X_val (Tensor): Validation data features
            y_val (Tensor): Validation data targets
            n_epochs (int): Amount of training process iterations
            init_lr (float): Learning rate initial value
            decay (float): Learning rate decay ratio
        '''
        n_samples = X_train.shape[0]
        n_classes = 10

        epsilon = 1e-10
        for epoch in range(n_epochs):
            epoch_loss = 0
            epoch_acc = []

            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                current_batch_size = X_batch.shape[0]
                if current_batch_size == 0:
                    continue

                y_batch_one_hot = np.zeros((current_batch_size, n_classes))
                y_batch_one_hot[np.arange(current_batch_size), y_batch] = 1.0

                y_pred = self.forward(inputs=X_batch)   

                # calculate loss
                loss = -np.mean(y_batch_one_hot * np.log(y_pred + epsilon)) # small epsilon to prevent log(0)

                # calculate accuracy
                acc = np.mean(np.argmax(y_pred, axis=1) == y_batch)
                epoch_acc.append(acc)

                # backpropagation
                output_grad = (y_pred -y_batch_one_hot) / current_batch_size
                t = epoch + 1
                learning_rate = init_lr / (1 + decay*epoch)
                grad_3 = self.layer3.backward(output_grad, learning_rate, t)
                grad_2 = self.layer2.backward(grad_3, learning_rate, t)
                grad_1 = self.layer1.backward(grad_2, learning_rate, t)

                epoch_loss += loss

            epoch_avg_acc = np.mean(epoch_acc)
            epoch_loss = epoch_loss / (n_samples / batch_size)
            self.loss_log.append(epoch_loss)
            self.train_acc_log.append(epoch_avg_acc)

            # predictions on validation data
            val_indices = np.random.permutation(X_val.shape[0])
            X_val_shuffled = X_val[val_indices]
            y_val_shuffled = y_val[val_indices]
            val_preds = self.predict(X_val_shuffled)
            val_acc = np.mean(val_preds == y_val_shuffled)
            self.val_acc_log.append(val_acc)

            print(f"Epoch: {epoch} || Loss: {epoch_loss:.5f} || Training Acc: {epoch_avg_acc*100:.2f}% || Validation Acc: {val_acc*100:.2f}%")

    def predict(self, X):
        y_pred_probs = self.forward(inputs=X)
        predictions = np.argmax(y_pred_probs, axis=1)
        return predictions