CONFIG = {
    "num_episodes": 16*500,
    "num_episode_steps": 500,
    "num_searches": 6,
    "max_depth": 1,
    "verbose": False,
    "training_interval": 32,
    "minibatch_size": 16,
    "discount_factor": 0.99,
    "exploration": 1,
    "game": "gym",
    "gym": {
        "env_name": "CartPole-v1",
        "w": 5,
        "q": 3
    },
    "dynamics_nn": {
        "input_dim": 8+1 ,
        "layers": [32]*1,
        "output_dim": 8 + 1,
        "activation_function": "sigmoid",
        "k": 1,
        "learning_rate": 0.00001
    },

    "prediction_nn": {
        "input_dim": 8,
        "layers": [32]*1,
        "output_dim": 3,
        "activation_function": "sigmoid",
        "k": 1,
        "learning_rate": 0.00001
    },


    "representation_nn": {
        "input_dim": 16,
        "layers": [32]*1,
        "output_dim": 8,
        "activation_function": "sigmoid",
        "k": 1,
        "learning_rate": 0.00001
    }
}
