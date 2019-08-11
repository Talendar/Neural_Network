"""
    By Talendar (Gabriel Nogueira)
"""

import numpy as np
import random
import time
import datetime


class NeuralNetwork:
    """
    """

    def __init__(self, layers_size=None, cost_type="mse", layers_activation="sigmoid", logs_path="./logs"):
        """
        Constructor
        :param layers_size: list containing the sizes of the layers. The layers activation functions will be the default one.
        """
        self.cost_type = cost_type
        self.layers = []
        self.logs_path = logs_path

        if layers_size is not None:
            self.layers.append(NeuralLayer(layers_size[0], input_count=0, activation="input_layer"))
            for s in layers_size[1:]:
                input_count = self.layers[-1].size
                self.layers.append(NeuralLayer(s, input_count, layers_activation))


    def add_layer(self, size, activation="sigmoid"):
        """
        :param size:
        :param activation:
        :return:
        """
        try:
            self.layers.append(NeuralLayer(size[0], input_count=0, activation="input_layer"))
            for s in size[1:]:
                input_count = self.layers[-1].size
                self.layers.append(NeuralLayer(s, input_count, activation))

        except TypeError:
            if len(self.layers) == 0:
                self.layers.append(NeuralLayer(size, input_count=0, activation="input_layer"))
            else:
                input_count = self.layers[-1].size
                self.layers.append(NeuralLayer(size, input_count, activation))


    def predict(self, x):
        """
        Wrapper for the feedforward function that uses the network's current weights and bias.

        :param x: column vector containing the features of the sample. If an out of shape numpy array is fed, this function
        won't work properly due to errors in matrix multiplications.
        :return:
        """
        weights = [l.weights for l in self.layers[1:]]
        bias = [l.bias for l in self.layers[1:]]

        return self.feedforward(weights, bias, x)


    def feedforward(self, weights, bias, x):
        """
        Feedforward.

        :param weights: list with the weights matrix for each layer (excluding the first layer).
        :param bias: list with the bias vector for each layer (excluding the first layer).
        :param x: column vector containing the features of the sample. If an out of shape numpy array is fed, this function
        won't work properly due to errors in matrix multiplications.
        :return: a vector (numpy array) containing the output of each neuron of the output layer.
        """
        a = self.colvector(x)
        for i, l in enumerate(self.layers[1:]):
            w, b = weights[i], bias[i]
            a = l.activate(np.dot(w, a) + b)

        return a


    def costfunc_unit(self, h, y, derivative=False):

        """
        :param x:
        :param y:
        :param derivative:
        :return:
        """
        if self.cost_type.lower() == "mse":
            if not derivative:
                return ((h - y) ** 2) / 2
            else:
                return h - y

        raise NameError("Cost function of the type \"%s\" is not defined!" % str(self.cost_type))


    def costfunc(self, data):
        """
        :param training_set:
        :param labels:
        :return:
        """
        p_cols = 1
        try:
            p_cols = len(data[0][0])
        except TypeError:
            pass

        l_cols = 1
        try:
            l_cols = len(data[0][1])
        except TypeError:
            pass

        m = len(data)
        predictions = np.zeros((m, p_cols))
        labels = np.zeros((m, l_cols))

        for i in range(m):
            h, y = data[i]
            predictions[i] = h.transpose()
            labels[i] = y.transpose()

        if self.cost_type.lower() == "mse":
            loss = np.sum((predictions - labels) ** 2) / (2 * m)
            return loss

        raise NameError("Cost function of the type \"%s\" is not defined!" % str(self.cost_type))

        # non-vectorized general implementation
        """total = 0
        for x, y in zip(training_set, labels):
            x, y = self.colvector(x), self.colvector(y)
            total += sum(self.costfunc_unit(self.predict(x), y))
        return total / len(training_set)"""


    def colvector(self, v):
        """
        Turn a numpy array into a column vector.
        :param v:
        :return:
        """
        try:
            v.shape = (len(v), 1)
        except Exception:
            pass

        return v


    def generate_mini_batches(self, training_data, labels, mini_batch_size):
        """
        :param training_data:
        :param mini_batch_size_pc:
        :return:
        """
        m = len(training_data)
        indexes = [n for n in range(m)]
        random.shuffle(indexes)

        mini_batches_x = [training_data[indexes[k:(k + mini_batch_size)]] for k in range(0, m, mini_batch_size)]
        mini_batches_y = [labels[indexes[k:(k + mini_batch_size)]] for k in range(0, m, mini_batch_size)]

        return mini_batches_x, mini_batches_y


    def backpropagation(self, x, y):
        """
        :param x:
        :param y:
        :return: a tuple containing, respectively: the gradient of the cost function with respect to the weights; the
        gradient of the cost function with respect to the bias; the prediction (activation result of the output layer)
        of the model for the given sample.
        """
        grad_w = [np.zeros(l.weights.shape) for l in self.layers[1:]]
        grad_b = [np.zeros(l.bias.shape) for l in self.layers[1:]]

        # FORWARD PASS
        activations = [self.colvector(x)]
        zs = []  # note that the input layer is not considered! So this list will have a size num_layers - 1

        for l in self.layers[1:]:  # excluding the input layer from the iteration
            w, b = l.weights, l.bias
            a = self.colvector(activations[-1])
            z = np.dot(w, a) + b

            zs.append(z)
            activations.append(l.activate(z))

        # BACKWARD PASS
        delta = self.costfunc_unit(activations[-1], y, derivative=True) * self.layers[-1].activate(zs[-1], derivative=True)  # initial delta is delta_L (delta for the output layer)

        # gradients with respect to the output layer
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].transpose())

        # gradients with respect to the hidden layers
        for l in range((len(self.layers) - 2), 0, -1):
            w_lp1, delta_lp1 = self.layers[l + 1].weights, delta
            a_der = self.layers[l].activate(zs[l - 1], derivative=True)  # l - 1 because zs doesnt have the input layer!

            delta = np.dot(w_lp1.transpose(), delta) * a_der
            grad_b[l - 1] = delta
            grad_w[l - 1] = np.dot(delta, activations[l - 1].transpose())

        return grad_w, grad_b, activations[-1]


    def compute_gradient_numerically(self, data_x, data_y):
        """
        Compute the gradient of the cost function numerically. To be used in gradient checking only.

        :param data_x:
        :param data_y:
        :return:
        """
        eps = 1e-5

        base_weights = [l.weights for l in self.layers[1:]]
        base_bias = [l.bias for l in self.layers[1:]]

        grad_w = [np.zeros(l.weights.shape) for l in self.layers[1:]]
        grad_b = [np.zeros(l.bias.shape) for l in self.layers[1:]]

        for l in range(0, len(self.layers) - 1):
            for row in range(len(base_weights[l])):

                # WEIGHTS
                for col in range(len(base_weights[l][row])):
                    temp_weights_plus = [w.copy() for w in base_weights]
                    temp_weights_minus = [w.copy() for w in base_weights]

                    temp_weights_plus[l][row][col] += eps
                    temp_weights_minus[l][row][col] -= eps

                    predictions_plus = [( self.feedforward(temp_weights_plus, base_bias, x), y )
                                        for x, y in zip(data_x, data_y)]
                    predictions_minus = [( self.feedforward(temp_weights_minus, base_bias, x), y )
                                        for x, y in zip(data_x, data_y)]
                    grad_w[l][row][col] = (self.costfunc(predictions_plus) - self.costfunc(predictions_minus)) / (2*eps)
                    #print("W[%d][%d][%d] = %f" % (l, row, col, grad_w[l][row][col]))

                # BIAS
                temp_bias_plus = [b.copy() for b in base_bias]
                temp_bias_minus = [b.copy() for b in base_bias]

                temp_bias_plus[l][row] += eps
                temp_bias_minus[l][row] -= eps

                predictions_plus = [( self.feedforward(base_weights, temp_bias_plus, x), y )
                                    for x, y in zip(data_x, data_y)]
                predictions_minus = [( self.feedforward(base_weights, temp_bias_minus, x), y )
                                     for x, y in zip(data_x, data_y)]
                grad_b[l][row] = (self.costfunc(predictions_plus) - self.costfunc(predictions_minus)) / (2*eps)
                #print("\nB[%d][%d] = %f\n" % (l, row, grad_b[l][row]))

        return grad_w, grad_b


    def regularization_term(self, m, method, factor, derivative=False, w=None):
        """

        :param m:
        :param method:
        :param factor:
        :param derivative:
        :param w:
        :return:
        """
        if method.lower() == "l2":
            if derivative:
                return w * factor / m
            else:
                w_sum = 0
                for l in self.layers:
                    w_sum += np.sum(np.square(l.weights))

                return w_sum * factor / (2 * m)

        raise NameError("Regularization method of the type \"%s\" is not defined!" % str(method))


    def sgd(self, training_data, labels, epochs, learning_rate, mini_batch_size,
            reg_method=None, reg_factor=0, verbose=True, gradient_checking=False):
        """
        Fit the network's parameters to the training data using Stochastic Gradient Descent.

        :param training_data: numpy ndarray containing, in each row, all the samples of the training set (each row is a
        vector of the features representing one of the samples).
        :param labels: column vector, in the form of a numpy array of the appropriate shape, containing the labels with
        respect to the training set samples.
        :param epochs: number of iterations to be ran by the algorithm.
        :param mini_batch_size: size of the mini batches of samples to be used in each iteration.
        :return:
        """
        # clearing log files
        if gradient_checking:
            with open(self.logs_path + "/log_gradient_checking.txt", "w") as file:
                file.write("< Started at: " + str(datetime.datetime.now()) + " >\n\n")

        # SGD
        starting_time = time.time()
        all_costs = []

        for e in range(epochs):
            mini_batches = self.generate_mini_batches(training_data, labels, mini_batch_size)
            predictions = []

            for mini_batch_x, mini_batch_y in zip(mini_batches[0], mini_batches[1]):

                # gradient of the cost function (considering the current batch) with respect to the weights and bias
                grad_w = [np.zeros(l.weights.shape) for l in self.layers[1:]]
                grad_b = [np.zeros(l.bias.shape) for l in self.layers[1:]]
                for x, y in zip(mini_batch_x, mini_batch_y):
                    x, y = self.colvector(x), self.colvector(y)
                    grad_w_variation, grad_b_variation, h = self.backpropagation(x, y)
                    predictions.append((h, y))  # save the values of the predictions and its associated label

                    grad_w = [cur + var for cur, var in zip(grad_w, grad_w_variation)]
                    grad_b = [cur + var for cur, var in zip(grad_b, grad_b_variation)]

                m = len(mini_batch_y)
                grad_w = [w/m for w in grad_w]
                grad_b = [b/m for b in grad_b]

                # updating weights and bias
                for i, l in enumerate(self.layers[1:]):
                    gw, gb = grad_w[i], grad_b[i]

                    l.bias -= (learning_rate * gb)
                    reg_term = 0 if reg_method is None else self.regularization_term(m, reg_method,
                                                                    reg_factor, derivative=True, w=l.weights)
                    l.weights -= learning_rate * (gw + reg_term)

                # gradient checking (log results)
                if gradient_checking:
                    num_grad_w, num_grad_b = self.compute_gradient_numerically(mini_batch_x, mini_batch_y)

                    flat_num_grad = np.concatenate((num_grad_w[0].flatten(), num_grad_b[0].flatten()))
                    flat_grad = np.concatenate((grad_w[0].flatten(), grad_b[0].flatten()))
                    for l in range(1, len(self.layers) - 1):
                        flat_num_grad = np.concatenate((flat_num_grad,
                                                         np.concatenate((num_grad_w[l].flatten(), num_grad_b[l].flatten()))))
                        flat_grad = np.concatenate((flat_grad,
                                                    np.concatenate((grad_w[l].flatten(), grad_b[l].flatten()))))

                    relative_error = np.linalg.norm(flat_grad - flat_num_grad) / np.linalg.norm(flat_grad + flat_num_grad)
                    with open(self.logs_path + "/log_gradient_checking.txt", "a") as file:
                        file.write("Relative error: %f\n\n" % relative_error)

            # cost function
            all_costs.append(self.costfunc(predictions))

            # print status
            if verbose:
                pc = (e + 1) / epochs
                bar = "|" + "#" * int(70 * pc) + "_" * int(70 * (1 - pc)) + "|"

                elapsed_time = time.time() - starting_time
                remaining_time = elapsed_time * (epochs - e + 1) / (e + 1)
                m, s = divmod(remaining_time, 60)
                h, m = divmod(m, 60)

                try:
                    # noinspection PyUnresolvedReferences
                    from IPython.display import clear_output  # required to clean the output when using notebooks
                    clear_output(wait=True)
                except ImportError:
                    print("\n" * 50)

                print("Epoch: %d/%d  %s  ETA: %02dh %02dmin %02ds"
                      "\nTotal loss/cost: %.6f"
                      % ((e + 1), epochs, bar, h, m, s, all_costs[-1]))

        return all_costs

    def save_params(self, file_name="./data/params.txt"):
        """
        :param file_name:
        :return:
        """
        file = open(file_name, "w")

        for i in range(1, len(self.layers)):
            l = self.layers[i]
            for W in l.weights:
                for w in W:
                    to_write = str(w).replace("[", "").replace("]", "")
                    file.write(to_write + " ")
                file.write("\n")

            for b in l.bias:
                to_write = str(b).replace("[", "").replace("]", "")
                file.write(to_write + " ")
            file.write("\n")

        file.close()


    def load_params(self, file_name="./data/params.txt"):
        """
        :param file_name:
        :return:
        """
        file = open(file_name, "r")
        for l in self.layers[1:]:

            for i in range(len(l.weights)):
                raw = file.readline().split()
                for j in range(len(l.weights[i])):
                    l.weights[i][j] = float(raw[j])

            raw = file.readline().split()
            for i in range(len(l.bias)):
                l.bias[i] = float(raw[i])
        file.close()


class NeuralLayer:
    """
    """

    def __init__(self, size, input_count, activation="sigmoid"):
        self.size = size
        self.input_count = input_count
        self.activation = activation

        if activation.lower() == "input_layer":
            self.weights, self.bias = None, None
        else:
            self.weights = np.random.uniform(low=-1, high=1, size=(size, input_count))
            self.bias = np.random.uniform(low=-1, high=1, size=(size, 1))


    def activate(self, z, derivative=False):
        """
        :param z:
        :return:
        """
        if self.activation.lower() == "input_layer":
            raise ValueError("Tried to activate the neurons from the input layer!")

        if self.activation.lower() == "sigmoid":
            return 1 / (1 + np.exp(-z)) if not derivative else (self.activate(z) * (1 - self.activate(z)))

        if self.activation.lower() == "relu":
            return np.maximum(z, 0) if not derivative else np.ceil(np.clip(z, 0, 1))

        if self.activation.lower() == "linear":
            return z if not derivative else 1

        raise NameError("Activation function of the type \"%s\" is not defined!" % str(self.activation))
