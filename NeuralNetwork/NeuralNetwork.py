import jax.numpy as jnp
from jax import random
import jax.nn as nn

from NeuralNetwork.NeuralNetworkLayer import NeuralNetworkLayer

key = random.PRNGKey(0)


class NeuralNetwork:
    hidden_layers: list[NeuralNetworkLayer]
    include_hidden_states: bool
    output_layer: NeuralNetworkLayer
    input_layer: NeuralNetworkLayer

    def __init__(self, config):
        input_dim = config['input_dim']
        layers = config['layers']
        output_dim = config['output_dim']
        activation_function = config['activation_function']
        self.k = config['k']
        self.learning_rate = config['learning_rate']

        key = config.get('key', random.PRNGKey(0)) 
        glorot = nn.initializers.glorot_normal()
        bias_init = nn.initializers.normal(stddev=0.01)

        key, input_key, *hidden_keys, output_key = random.split(key, len(layers) + 2)
        
        self.input_layer = NeuralNetworkLayer(
            n_neurons=layers[0],
            activation_function=activation_function,
            params={
                "weights": glorot(input_key, (layers[0], input_dim)),
                "bias": bias_init(input_key, (layers[0],))
            }
        )

        self.hidden_layers = []
        for i, (n, in_dim) in enumerate(zip(layers[1:], layers[:-1])):
            key = hidden_keys[i]
            self.hidden_layers.append(
                NeuralNetworkLayer(
                    n_neurons=n,
                    activation_function=activation_function,
                    params={
                        "weights": glorot(key, (n, in_dim)),
                        "hidden_weights": glorot(key, (n, n)), 
                        "bias": bias_init(key, (n,))
                    }
                )
            )

        # Output layer
        self.output_layer = NeuralNetworkLayer(
            n_neurons=output_dim,
            activation_function="identity",
            params={
                "weights": glorot(output_key, (output_dim, layers[-1])),
                "hidden_weights": jnp.zeros((output_dim,)), 
                "bias": bias_init(output_key, (output_dim,))
            }
        )
        self.layer_parameters = [self.input_layer.parameters] + \
                                [layer.parameters for layer in self.hidden_layers] + \
                                [self.output_layer.parameters]

    def loss_function(self, activation, targets):
        return 0.5 * jnp.sum((activation - targets) ** 2)

    def forward(self, input, params=None):
        if params is None:
            params = self.layer_parameters
        current_input = self.input_layer.compute_output(input, params[0])

        for i, layer in enumerate(self.hidden_layers):
            new_state = layer.compute_output(
                current_input, params[i + 1])
            current_input = new_state

        output = self.output_layer.compute_output(
            current_input, params[-1])

        return output
