import optuna
import warnings
from system import *

warnings.filterwarnings("ignore")

def eval_params(params):
    try:
        system = System(params)
        score = system.train()
        return score
    except:
        return -1



def objective(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-3)
    discount_factor = trial.suggest_uniform("discount_factor", 0.90, 0.99)
    exploration = trial.suggest_uniform("exploration", 0.1, 5.0)

    config = CONFIG.copy()
    config["discount_factor"] = discount_factor
    config["exploration"] = exploration  

    config["dynamics_nn"]["learning_rate"] = learning_rate
    config["prediction_nn"]["learning_rate"] = learning_rate
    config["representation_nn"]["learning_rate"] = learning_rate

    score = eval_params(config)
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best trial:")
trial = study.best_trial
print(f"  Score: {trial.value}")
print("  Hyperparameters: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
