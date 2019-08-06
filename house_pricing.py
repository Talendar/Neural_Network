import pandas as pd
import numpy as np
from neural_network import NeuralNetwork, NeuralLayer


def validate(net, testing_set, testing_labels, accepted_error_pc):
    hits = 0
    for x, y in zip(testing_set, testing_labels):
        h = net.predict(x)
        if (y*(1 + accepted_error_pc)) >= h >= (y*(1 - accepted_error_pc)):
            hits += 1

    return hits / len(testing_set)


if __name__ == "__main__":
    # loading and pre-processing data
    data = pd.read_csv("./data/kc_house_data.csv")
    data.drop(["id", "date", "waterfront", "view", "yr_renovated", "zipcode"], axis=1, inplace=True)

    data = (data - data.mean()) / data.std()  # normalization
    #data = data.sample(frac=0.6)

    testing_set = data.sample(frac=0.2)
    training_set = data.drop(testing_set.index)

    testing_set, testing_labels = testing_set.drop("price", axis=1).values, testing_set["price"].values
    training_set, training_labels = training_set.drop("price", axis=1).values, training_set["price"].values

    # fitting
    net = NeuralNetwork([14, 64, 64, 64, 1], layers_activation="sigmoid")
    net.layers[-1].activation = "linear"

    net.sgd(training_set, training_labels, epochs=100, learning_rate=0.1, mini_batch_size=32)

    accuracy = validate(net, testing_set, testing_labels, 0.1)
    print("Accuracy: %.2f%%" % (accuracy*100))

