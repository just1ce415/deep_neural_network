import numpy as np
from custom_functions import relu, sigmoid, relu_backward, sigmoid_backward
import json
import copy

class DeepNN:
    def __init__(self, X_train, Y_train, X_test, Y_test, layer_dims):
        """
        X_train - 2D numpy array of shape (num_of_feautures, num_of_training_examples)
        Y_train - 2D numpy array of shape (1, num_of_training_examples)
        X_test - 2D numpy array of shape (num_of_feautures, num_of_test_examples)
        Y_test - 2D numpy array of shape (1, num_of_test_examples)
        layer_dims - and array with input size + number of neurons on each (hidden) layer
        + output size
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.layer_dims = layer_dims
        self.parameters = {}


    def __initialize(self):
    
        parameters = {}
        L = len(self.layer_dims) # number of layers in the network

        for l in range(1, L):
            parameters["W" + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) * 0.1#np.sqrt(1/2)
            parameters["b" + str(l)] = np.zeros((self.layer_dims[l], 1))
        
            assert(parameters['W' + str(l)].shape == (self.layer_dims[l], self.layer_dims[l - 1]))
            assert(parameters['b' + str(l)].shape == (self.layer_dims[l], 1))

        
        return parameters


    def __linear_forward(self, A, W, b):
    
        Z = np.dot(W, A) + b
        
        # Normalization
        eps = 0.0001
        means = np.mean(Z, axis=1, keepdims=True)
        stds = np.std(Z, axis=1, keepdims=True)
        Z = (Z - means) / (stds + eps)

        cache = (A, W, b)
    
        return Z, cache


    def __linear_activation_forward(self, A_prev, W, b, activation):
    
        if activation == "sigmoid":
            Z, linear_cache = self.__linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)
        
        elif activation == "relu":
            Z, linear_cache = self.__linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)
        
        cache = (linear_cache, activation_cache)

        return A, cache

    def __forward_prop(self, X, parameters):

        caches = []
        A = X
        L = len(parameters) // 2
    
        for l in range(1, L):
            A_prev = A 
            A, cache = self.__linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], 'relu')
            caches.append(cache)
    
        AL, cache = self.__linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], 'sigmoid')
        caches.append(cache)
          
        return AL, caches


    def __get_loss(self, AL, Y):
        m = Y.shape[1]
        cost = -1/m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        cost = np.squeeze(cost)
    
        return cost


    def __linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1/m * np.dot(dZ, A_prev.T)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
    
        return dA_prev, dW, db


    def __linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
    
        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.__linear_backward(dZ, linear_cache)
        
        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.__linear_backward(dZ, linear_cache)
    
        return dA_prev, dW, db


    def __backward_prop(self, AL, Y, caches):
        grads = {}
        L = len(caches) # the number of layers
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
        current_cache = caches[L - 1]
        dA_prev_temp, dW_temp, db_temp = self.__linear_activation_backward(dAL, current_cache, "sigmoid")
        grads["dA" + str(L-1)] = dA_prev_temp
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp
    
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.__linear_activation_backward(dA_prev_temp, current_cache, "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
        
        return grads

    def __update_parameters(self, params, grads, learning_rate):
        parameters = params.copy()
        L = len(parameters) // 2

        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

        return parameters


    def fit(self, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
        costs = []                         # keep track of cost
    
        parameters = self.__initialize()
    
        for i in range(0, num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = self.__forward_prop(self.X_train, parameters)
        
            # Compute cost.
            cost = self.__get_loss(AL, self.Y_train)
        
            # Backward propagation.
            grads = self.__backward_prop(AL, self.Y_train, caches)
        
            # Update parameters.
            parameters = self.__update_parameters(parameters, grads, learning_rate)
        
            # Print the cost every 100 iterations
            if print_cost and i % 100 == 0 or i == num_iterations - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0 or i == num_iterations:
                costs.append(cost)
    
        self.parameters = parameters
        return parameters, costs


    def __predict(self, X):
        Y_hat, _ = self.__forward_prop(X, self.parameters)
        for i in range(X.shape[1]):
            if Y_hat[0, i] > 0.5:
                Y_hat[0, i] = 1
            else:
                Y_hat[0, i] = 0
        return Y_hat


    def validate(self):
        Y_prediction_test = self.__predict(self.X_test)
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - self.Y_test)) * 100))
        return Y_prediction_test


    def train_validate(self):
        Y_prediction_train = self.__predict(self.X_train)
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - self.Y_train)) * 100))
        return Y_prediction_train

    def save_params(self, name):
        dic = copy.deepcopy(self.parameters)
        for key, val in dic.items():
            dic[key] = val.tolist()
        with open(name, 'w', encoding='utf-8') as fptr:
            json.dump(dic, fptr)

    def load_params(self, path):
        with open(path, 'r') as fptr:
            self.parameters = json.load(fptr)
        for key, val in self.parameters.items():
            self.parameters[key] = np.asarray(val)