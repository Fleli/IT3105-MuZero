import numpy as np


class NeuralNetworkLayer:
    
    is_input_layer = False
    
    activation_function: any  # sigmoid, etc.
    
    n_neurons: int
    
    previous_layer: any
    
    _ACTIVATION_FUNCTIONS = {
        "sigmoid": lambda x: 1 / (1 + np.e ** (-x)),
        "identity": lambda x: x,
        "relu": lambda x: max(x, 0),
        "tanh": lambda x: (np.e ** x - 1) / (2 * np.e ** x)
    }

    def __init__(self, n_neurons, previous_layer, activation_function, include_bias: bool):
        
        self.n_neurons = n_neurons
        if isinstance(activation_function, str):
            self.activation_function = self._ACTIVATION_FUNCTIONS[activation_function]
        else:
            self.activation_function = activation_function
        
        if previous_layer is None:
            self.is_input_layer = True
            return
        
        self.include_bias = include_bias
        self.previous_layer = previous_layer
        
    def compute_output(self, input: list[float], params) -> float:
        
        if self.is_input_layer:
            return input
        
        weights = {
            (from_neuron, to_neuron) : params[ from_neuron + to_neuron * self.n_neurons ]
            for from_neuron in range(self.previous_layer.n_neurons)
            for to_neuron in range(self.n_neurons)
        }
        
        if self.include_bias:
            b0 = self.n_neurons * self.previous_layer.n_neurons
            biases = params[b0 : b0 + self.n_neurons]
            output_of_previous = self.previous_layer.compute_output(input, params[b0 + self.n_neurons : ])
        else:
            output_of_previous = self.previous_layer.compute_output(input, params[self.n_neurons : ])
        
        values = []
        for neuron in range(self.n_neurons):        # Skift ut med matrisemultiplikasjon
            values.append(biases[neuron] if self.include_bias else 0)
            for previous in range(self.previous_layer.n_neurons):
                values[neuron] += weights[previous, neuron] * output_of_previous[previous]
            values[neuron] = self.activation_function( values[neuron] )
        
        return values
