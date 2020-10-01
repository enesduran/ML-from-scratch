import numpy as np
import sys

class NeuralNet:
    def __init__(self, neuron_numbers_for_each_layer, activation_function, learning_rate, bias_value):
        """
            param layer_numbers: list: layer numbers including input and the output dimensions
            param activation_function: string : indicates non linear activation function
            param learning rate : float : learning rate of the model
        """
        super(NeuralNet, self).__init__()
        self.layer_values = []
        self.neuron_numbers_for_each_layer = neuron_numbers_for_each_layer
        self.activation_func = activation_function
        self.lr = learning_rate
        self.bias_value = bias_value
        self.weights = dict()
        self.derivative_values = dict()
        self.bias = dict()
        # initialize all the weights to 1
        for i in range(len(neuron_numbers_for_each_layer) - 1):
            self.weights["layer_" + str(i)] = np.random.normal(0, 0.01, [neuron_numbers_for_each_layer[i + 1], neuron_numbers_for_each_layer[i] + 1])
            self.derivative_values["layer_" + str(i)] = np.zeros([neuron_numbers_for_each_layer[i + 1], neuron_numbers_for_each_layer[i] + 1])

    def forward(self, input_vector, end_layer=None):
        """
        :param input_vector: numpy array of input
        :param end_layer : the layer which forward pass stops 
        :return: output: the output in numpy array
        """
        assert type(input_vector) == np.ndarray, "Inputs should be numpy arrays!"
        output = input_vector
        self.layer_values = []
        self.layer_values.append(output)
        # adding bias
        if end_layer is None: end_layer = len(self.neuron_numbers_for_each_layer)-1
             
        # add bias
        for i in range(end_layer):
            output = np.append(output, self.bias_value)
            output = self.activation(np.matmul(self.weights["layer_" + str(i)], output))
            self.layer_values.append(output)
        return output

    # updates the weights using mse loss
    def backward(self, y, X):
        """
        :param y: value of the label
        :param o: output of the NN coming from the forward() method
        :return: void
        """
        assert len(self.neuron_numbers_for_each_layer) >= 2 , "There is no layer"
        # initial derivative value is the derivative of loss wrt output and it will be updated throughout each pass
        o = self.forward(X)
        delta = 0
        for i in reversed(range(len(self.neuron_numbers_for_each_layer))):
            # check if last layer
            if i == len(self.neuron_numbers_for_each_layer)-1:
                delta = (self.activation_function_derivative(o) * (o - y))
            elif i == 0:    
                o_current = np.append(self.bias_value, X)
                layer_error =  (o_current[:,np.newaxis] * delta).T
                self.derivative_values["layer_" + str(i)] = layer_error
            else:    
                pass
        return

    def step(self):
        """updates the weights according to the value of derivatives and learning rate"""
        for i in range(len(self.neuron_numbers_for_each_layer) - 1):
            self.weights['layer_' + str(i)] -= self.derivative_values['layer_' + str(i)] * self.lr
        return

    def activation(self, linear_input):
        flag = (type(linear_input) == np.float64)
        #assert type(linear_input) == np.ndarray, "Type of the input is :"+str(type(linear_input))
        """returns the result of the inputs fed into non-linear function in a numpy array"""
        if self.activation_func == "relu":
            if not flag:
                nonlinear_output = np.array([max(0, element) for element in linear_input])
            else:
                nonlinear_output = max(0, linear_input)
        elif self.activation_func == "step":
            if not flag:
                nonlinear_output = np.array([element >= 0 for element in linear_input])
            else: 
                nonlinear_output = (linear_input >= 0)
        # sigmoid
        else:
            #nonlinear_output = np.ones_like(linear_input) / (1 + np.exp(-linear_input))
            nonlinear_output = 1 / (1 + np.exp(-linear_input))
        return nonlinear_output

    def mse_loss(self, y, o):
        """
        :param y: 1D input array
        :param o: 1D input array
        :return: scalar indicating mean square error
        """
        assert type(y) == np.ndarray and type(o) == np.ndarray and y.size == o.size, str(type(y))+" "+str(type(o)) 
        return np.square((y - o)).mean()

    def activation_function_derivative(self, layer):
        """
        takes the derivative of the layer with respect to the non-linear activation function
        :param layer: numpy array after non-linear activation function applied
        :return: derivation : numpy array of the values depending on the type of the function
        """
        assert type(layer) == np.ndarray, "Input ought to be a numpy array instance"
        if self.activation_func == "step":
            derivation = np.array([element > 0 for element in layer])
        elif self.activation_func == "sigmoid":
            derivation = layer * (1 - layer)
        # ReLU
        else:
            derivation = np.zeros(len(layer))
            derivation[np.where(layer > 0)] = 1
        return derivation

    def weight_assignment(self, weights, layer_number):
        self.weights["layer_" + str(layer_number)] = weights
        return 