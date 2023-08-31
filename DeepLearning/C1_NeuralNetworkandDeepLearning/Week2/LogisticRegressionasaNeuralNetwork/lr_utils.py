import numpy as np
import h5py

def load_dataset():
    train_dataset = h5py.File('E:/DeepLearning/C1_NeuralNetworkandDeepLearning/Week2/LogisticRegressionasaNeuralNetwork/datasets/train_catvnoncat.h5', 'r')
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('E:/DeepLearning/C1_NeuralNetworkandDeepLearning/Week2/LogisticRegressionasaNeuralNetwork/datasets/test_catvnoncat.h5', 'r')
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes



def check_hdf5_file(file_path):
    try:
        with h5py.File(file_path, 'r') as hdf5_file:
            print("HDF5 file is valid and can be opened successfully.")
            # You can also print out more information about the file's contents if needed
            print("Dataset keys:", list(hdf5_file.keys()))
    except Exception as e:
        print("Error:", e)
        print("HDF5 file is invalid or cannot be opened.")

# Specify the paths to your HDF5 files
train_hdf5_path = 'E:/DeepLearning/C1_NeuralNetworkandDeepLearning/Week2/LogisticRegressionasaNeuralNetwork/datasets/train_catvnoncat.h5'
test_hdf5_path = 'E:/DeepLearning/C1_NeuralNetworkandDeepLearning/Week2/LogisticRegressionasaNeuralNetwork/datasets/test_catvnoncat.h5'

# Check the validity of the HDF5 files
check_hdf5_file(train_hdf5_path)
check_hdf5_file(test_hdf5_path)
