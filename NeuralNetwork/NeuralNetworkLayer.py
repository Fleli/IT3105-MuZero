import jax.numpy as jnp

class NeuralNetworkLayer:
    
    is_input_layer = False
    activation_function: any
    n_neurons: int
    previous_layer: any

    def __init__(self, n_neurons, activation_function, params):
        self.parameters = params
        self.n_neurons = n_neurons

        if isinstance(activation_function, str):
            if activation_function == "sigmoid":
                self.activation_function = lambda x: 1 / (1 + jnp.exp(-x))
            elif activation_function == "relu":
                self.activation_function = lambda x: jnp.maximum(x, 0)
            elif activation_function == "tanh":
                self.activation_function = jnp.tanh
            elif activation_function == "identity":
                self.activation_function = lambda x: x
            else:
                raise ValueError("Unsupported activation function")
        else:
            self.activation_function = activation_function

    def compute_output(self, input: jnp.ndarray, params = None) -> jnp.ndarray:

        if params is None:
            params = self.parameters
        z = jnp.dot(params["weights"], input) 
        if params.get("bias", None) is not None:
            z = z + params["bias"]
        return self.activation_function(z)
