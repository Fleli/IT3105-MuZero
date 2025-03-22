CONFIG = {
    "num_episodes": 100000,
    "num_episode_steps": 100,
    "num_searches": 20,
    "max_depth": 10,
    "training_interval": 10,
    "minibatch_size": 64,
    
    "game": "gym",
    "gym": {
        "env_name": "CartPole-v1",
        "w": 5,   
        "q": 0
    },
    "dynamics_nn": {
        "input_dim": 4 + 1,  
        "layers": [5,5,5],
        "output_dim": 4 + 1,  
        "activation_function": "tanh",
        "include_bias": True,
    },
    
    "prediction_nn": {
        "input_dim":  4,
        "layers": [5,5,5],
        "output_dim": 3,
        "activation_function": "tanh",
        "include_bias": True,
    },
    
    
    "representation_nn": {
        "input_dim": 4,
        "layers": [5,5,5],
        "output_dim": 4,  
        "activation_function": "tanh",
        "include_bias": True,
    }
}
