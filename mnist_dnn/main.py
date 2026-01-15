import numpy as np

from layer import DenseLayer
from dnn import DenseNeuralNetwork

from mnist_loader import load_data, load_data_wrapper

if __name__ == "__main__":   
    np.random.seed(42)

    INPUT_SIZE = 784
    HIDDEN_SIZE = (256, 256)
    OUTPUT_SIZE = 10 

    (training_data, validation_data, test_data) = load_data()
    X_train, y_train = training_data
    X_test, y_test = test_data

    X_val, y_val = validation_data

    dnn = DenseNeuralNetwork(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, hidden_size=HIDDEN_SIZE)
    dnn.fit(X_train, y_train, init_lr = 0.001, decay = 0.001, n_epochs = 20, batch_size=256)