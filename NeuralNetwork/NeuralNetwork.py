import jax
import jax.numpy as jnp
import jax.random as random
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
        self.input_layer = NeuralNetworkLayer(
            n_neurons=layers[0],
            activation_function=activation_function,
            params={
                "weights": random.normal(key, (layers[0], input_dim)) * 0.1,
                "bias": random.normal(key, (layers[0],)) * 0.00001
            }
        )
        self.hidden_layers = [
            NeuralNetworkLayer(
                n_neurons=n,
                activation_function=activation_function,
                params={
                    "weights": random.normal(key, (n, in_dim)) * 0.1,
                    "hidden_weights": random.normal(key, (n, n)) * 0.1,
                    "bias": random.normal(key, (n,)) * 0.00000
                }
            )
            for n, in_dim in zip(layers[1:], layers[:-1])
        ]

        self.output_layer = NeuralNetworkLayer(
            n_neurons=output_dim,
            activation_function="identity",
            params={
                "weights": random.normal(key, (output_dim, layers[-1])) * 0.1,
                # typically no recurrent connection in output.
                "hidden_weights": jnp.zeros((output_dim,)),
                "bias": random.normal(key, (output_dim,)) * 0.00001
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
        hidden_states = []

        for i, layer in enumerate(self.hidden_layers):
            new_state = layer.compute_output(
                current_input, params[i + 1])
            hidden_states.append(new_state)
            current_input = new_state

        output = self.output_layer.compute_output(
            current_input, params[-1])

        return output
