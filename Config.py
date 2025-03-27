CONFIG = {
    "num_episodes": 100000,
    "num_episode_steps": 500,
    "num_searches": 20,
    "max_depth": 3,
    "training_interval": 10,
    "minibatch_size": 64,

    "game": "gym",
    "gym": {
        "env_name": "CartPole-v1",
        "w": 5,
        "q": 0
    },
    "dynamics_nn": {
        "input_dim": 4+1 ,
        "layers": [16]*3,
        "output_dim": 4 + 1,
        "activation_function": "relu",
        "k": 1,
        "learning_rate": 0.001
    },

    "prediction_nn": {
        "input_dim":  4+1,
        "layers": [16]*3,
        "output_dim": 3,
        "activation_function": "relu",
        "k": 1,
        "learning_rate": 0.001
    },


    "representation_nn": {
        "input_dim": 4,
        "layers": [16]*3,
        "output_dim": 4,
        "activation_function": "relu",
        "k": 1,
        "learning_rate": 0.001
    }
}
