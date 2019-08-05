from neural_network import NeuralNetwork
import numpy as np
import pandas as pd
from PIL import Image


def vectorize_labels(labels, vector_size):
    """

    :param y:
    :return:
    """
    m = len(labels)
    vectorized = np.zeros((m, vector_size))

    if vector_size > 1:
        vectorized = np.zeros((m, vector_size))

        for i in range(0, m):
            v = np.zeros(vector_size)
            v[labels[i]] = 1
            vectorized[i] = v
    else:
        labels.shape = (m, 1)
        vectorized = labels

    return vectorized


def visualize_image(image_array, width, height):
    """

    :param image_array:
    :param width:
    :param height:
    :return:
    """
    image_array.shape = (width, height)
    Image.fromarray(image_array.astype("uint8"), "L").show()


def validate(net, test_set, test_set_labels):
    """

    :param net
    :param test_set:
    :param test_set_labels:
    :return:
    """
    hits = 0
    for x, y in zip(test_set, test_set_labels):
        h = np.argmax(net.predict(x))
        hits += 1 if h == y else 0

    return hits / len(test_set)


# MAIN
if __name__ == "__main__":

    # training
    data = pd.read_csv("./data/mnist_train.csv")
    training_data, training_labels = data.drop("label", 1).values, vectorize_labels(data["label"].values, 10)

    #training_data = training_data[0:10000]
    #training_labels = training_labels[0:10000]

    net = NeuralNetwork([784, 16, 16, 10])
    net.sgd(training_data, training_labels, epochs=50, learning_rate=0.3, mini_batch_size=32)
    net.save_params()

    # validating
    data2 = pd.read_csv("./data/mnist_test.csv")
    test_data, test_labels = data2.drop("label", 1).values, data2["label"].values

    accuracy = validate(net, test_data, test_labels)
    print("\nAccuracy: %.2f%%" % (accuracy*100))
