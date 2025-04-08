CONFIG = {
    "num_episodes": 16*500,
    "num_episode_steps": 500,
    "num_searches": 10,
    "max_depth": 1,
    "verbose": False,
    "training_interval": 16,
    "minibatch_size": 16,
    "discount_factor": 0.95,
    "exploration": 3,
    "game": "gym",
    "gym": {
        "env_name": "CartPole-v1",
        "w": 5,
        "q": 0
    },
    "dynamics_nn": {
        "input_dim": 4+1 ,
        "layers": [256]*4,
        "output_dim": 4 + 1,
        "activation_function": "relu",
        "k": 1,
        "learning_rate": 0.00001
    },

    "prediction_nn": {
        "input_dim":  4+1,
        "layers": [256]*4,
        "output_dim": 3,
        "activation_function": "relu",
        "k": 1,
        "learning_rate": 0.00001
    },


    "representation_nn": {
        "input_dim": 4,
        "layers": [256]*4,
        "output_dim": 4,
        "activation_function": "relu",
        "k": 1,
        "learning_rate": 0.00001
    }
}
