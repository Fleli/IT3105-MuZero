CONFIG = {
    "num_episodes": 16*500,
    "num_episode_steps": 500,
    "num_searches": 6,
    "max_depth": 1,
    "verbose": False,
    "training_interval": 8,
    "minibatch_size": 4,
    "discount_factor": 1,
    "exploration": 1,
    "game": "gym",
    "gym": {
        "env_name": "CartPole-v1",
        "w": 5,
        "q": 5
    },
    "dynamics_nn": {
        "input_dim": 16+1 ,
        "layers": [32]*2,
        "output_dim": 16 + 1,
        "activation_function": "sigmoid",
        "k": 1,
        "learning_rate": 0.0001
    },

    "prediction_nn": {
        "input_dim": 16,
        "layers": [32]*2,
        "output_dim": 3,
        "activation_function": "sigmoid",
        "k": 1,
        "learning_rate": 0.0001
    },


    "representation_nn": {
        "input_dim": 24,
        "layers": [32]*2,
        "output_dim": 16,
        "activation_function": "sigmoid",
        "k": 1,
        "learning_rate": 0.0001
    }
}
