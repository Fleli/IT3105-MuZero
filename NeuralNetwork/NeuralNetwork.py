import jax
import jax.numpy as jnp
from NeuralNetwork.NeuralNetworkLayer import NeuralNetworkLayer

class NeuralNetwork:
    hidden_layers: list[NeuralNetworkLayer]
    include_bias: bool
    output_layer: NeuralNetworkLayer

    def __init__(self, input_dim: int, layers: list[int], output_layer: int, activation_function, include_bias: bool, max_output: float):
        layer_input_dims = [input_dim] + layers[:-1]
        self.hidden_layers = [
            NeuralNetworkLayer(
                n_neurons=n,
                activation_function=activation_function,
                params={ 
                    "weights": jnp.zeros((n, in_dim)),         
                    "hidden_weights": jnp.zeros((n, n)),          
                    "bias": jnp.zeros((n,)) if include_bias else None
                },
                include_bias=include_bias
            )
            for n, in_dim in zip(layers, layer_input_dims)
        ]
        self.max_output = max_output
        self.include_bias = include_bias
        
        self.output_layer = NeuralNetworkLayer(
            n_neurons=output_layer,
            activation_function=activation_function,
            params={
                "weights": jnp.zeros((output_layer, layers[-1])),  
                "bias": jnp.zeros((output_layer,)) if include_bias else None
            },
            include_bias=include_bias
        )

        self.layer_parameters = [layer.parameters for layer in self.hidden_layers] + [self.output_layer.parameters]

    def loss_function(self, activation, targets):
        return 0.5 * jnp.sum((activation - targets) ** 2)

    def forward(self, input, state_list):
        hidden_states = []
        current_input = input
        for layer, state in zip(self.hidden_layers, state_list):
            new_state = layer.compute_output(current_input, state)
            hidden_states.append(new_state)
            current_input = new_state
        output = self.output_layer.compute_output(current_input)
        return output, hidden_states


    def backward(self, grad_output, grad_state_next, stored_state, input):
        def forward_fn(layer_params, state):
            for i, layer in enumerate(self.hidden_layers):
                layer.parameters = layer_params[i]
            self.output_layer.parameters = layer_params[-1]
            output, new_states = self.forward(input, state)
            return output, new_states

        _, vjp_fn = jax.vjp(forward_fn, self.layer_parameters, stored_state)
        combined_grad = (grad_output, grad_state_next)
        grad_layer_parameters, grad_state = vjp_fn(combined_grad)
        return grad_layer_parameters, grad_state

    def BPTT(self, inputs, targets, initial_state, learning_rate, k):
        T = len(inputs)  # Number of time steps in the series

        state = initial_state
        stored_states = [state]
        stored_activations = []
        stored_losses = []

        grad_loss_fn = jax.grad(self.loss_function, argnums=0)

        for t in range(0, T - 1):
            output, state = self.forward(inputs[t], state)
            activation = output
            loss = self.loss_function(activation, targets[t])
            stored_activations.append(activation)
            stored_losses.append(loss)
            stored_states.append(state)

            if ((t + 1) % k == 0) or (t == T - 1):
                grad_state = 0
                grad_layer_parameters = jax.tree_map(jnp.zeros_like, self.layer_parameters)

                for j in range(t, max(0, t - k + 1), -1):
                    grad_output = grad_loss_fn(stored_activations[j], targets[j])
                    grad_param_j, grad_state = self.backward(grad_output, grad_state, stored_states[j], inputs[j])
                    grad_layer_parameters = jax.tree_map(lambda g1, g2: g1 + g2,
                                                         grad_layer_parameters, grad_param_j)

                self.layer_parameters = jax.tree_map(lambda param, grad: param - learning_rate * grad,
                                                     self.layer_parameters, grad_layer_parameters)
                for i, layer in enumerate(self.hidden_layers):
                    layer.parameters = self.layer_parameters[i]
                self.output_layer.parameters = self.layer_parameters[-1]

                stored_states = [state]
                stored_activations = []
                stored_losses = []
