# check which model script is best
"""
From: https://towardsdatascience.com/state-of-the-art-machine-learning-hyperparameter-optimization-with-optuna-a315d8564de1


Optuna names its optimization algorithms into two different categories, namely the sampling strategy and the pruning strategy.

    Sampling strategy: algorithms that select the best parameter combination by concentrating on areas where hyperparameters are giving better results.
    Pruning strategy: Early-stopping-based optimization methods as we discussed above.
"""

"""Nested cross-validation for model comparison"""

# TODO:
# - improve model configuration
# - seperate hyperparameters from training parameters
# - also use different config format
# - add more models
# - add lr scheduler, early stopping (?), etc. pp. -> HyperbandPruner
# - parallelize optuna trials?


import logging
from datetime import datetime

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.models.scalers_3d import StandardScaler3D

LEVEL = logging.DEBUG
logging.basicConfig(
    level=LEVEL,
    format="[%(asctime)s] [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)  # , logging.FileHandler('training_log.log')])

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)
is_cuda_available = device == torch.device("cuda")
logging.debug(f"Using device: {device}")

# Define Constants
BATCH_SIZE = 32  # not used, optimized
EPOCHS = 5
K_FOLDS = 3


def time_stamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out  # BCEWithLogitsLoss will apply sigmoid


class CNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CNN, self).__init__()
        self.layer1 = nn.Conv1d(input_size, hidden_size, kernel_size=3)
        self.layer2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()


# Define a dictionary of models with their respective hyperparameters and native data format
models_dict = {
    "MLP": {
        "class": MLP,
        "hyperparameters": {
            "hidden_size": {"type": "int", "low": 128, "high": 1024},
            "lr": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
            "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
        },
    },
    "CNN": {
        "class": CNN,
        "hyperparameters": {
            "hidden_size": {"type": "int", "low": 128, "high": 1024},
            "lr": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
            "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
        },
    },
    # Define other models with their respective hyperparameters and input sizes
}


def initialize_model(model_name, input_size, **hyperparams):
    model_class = models_dict[model_name]["class"]
    # Extracting lr and batch_size from hyperparams and not passing them to the model's constructor
    lr = hyperparams.pop("lr")  # finally a use for pop
    batch_size = hyperparams.pop("batch_size")
    model = model_class(input_size=input_size, **hyperparams).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer


def get_input_size(model_name, X):
    data_format = models_dict[model_name]["format"]
    match data_format:
        case "2D":
            return X.shape[2] * X.shape[1]
        case "3D":
            return X.shape[2]
        case _:
            raise ValueError(f"Unknown data format: {data_format}")


def load_dataset():
    X = np.load("data/data.npy")
    y = np.load("data/target.npy")
    return X, y


def create_dataloaders(
    X_train, y_train, X_test, y_test, batch_size, is_validation=False
):
    if not len(X_train.shape) == len(X_test.shape) == 3:
        raise ValueError(
            "X_train and X_test must have 3 dimensions: (samples, timesteps, features)"
        )

    scaler = StandardScaler3D()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_data = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1)
    )
    test_data = TensorDataset(
        torch.FloatTensor(X_test), torch.FloatTensor(y_test).view(-1, 1)
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if not is_cuda_available else 4,
        pin_memory=is_cuda_available,
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if not is_cuda_available else 4,
        pin_memory=is_cuda_available,
    )

    data_set = "Validation" if is_validation else "Test"
    logging.debug(
        f"Train Data: {len(train_data)} samples, {data_set} Data: {len(test_data)} samples"
    )

    return train_loader, test_loader


def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs, is_test=False
):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)

        phase = "Final Training" if is_test else "Training"
        data_set = "Test" if is_test else "Validation"
        logging.debug(
            f"{phase} | Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, {data_set} Loss: {val_loss:.4f}, {data_set} Accuracy: {val_accuracy:.4f}"
        )


def evaluate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred_logits = model(X_batch)
            loss = criterion(y_pred_logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            # Metric = Accuracy
            y_pred_classes = (torch.sigmoid(y_pred_logits) >= 0.5).float()
            total += y_batch.size(0)
            correct += (y_pred_classes == y_batch).sum().item()

    average_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return average_loss, accuracy


def cross_validate(X, y, model_name, input_size, k_folds, **hyperparams):
    kfold = KFold(n_splits=k_folds, shuffle=True)
    val_losses = []

    for fold, (train_index, val_index) in enumerate(kfold.split(X, y)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        train_loader, val_loader = create_dataloaders(
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=hyperparams["batch_size"],
            is_validation=True,
        )
        model, criterion, optimizer = initialize_model(
            model_name, input_size, **hyperparams
        )
        train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)
        average_loss, accuracy = evaluate_model(model, val_loader, criterion)
        val_losses.append(average_loss)

        logging.info(
            f"Fold {fold} | Accuracy: {accuracy:.2f} | Validation Loss: {average_loss:.2f},"
        )
    return val_losses


def create_objective_function(X, y, model_name, input_size, model_info):
    # functional programming: create a closure to capture X, y, and input_size
    def objective(trial):
        """
        The 'objective' function can access 'X', 'y', and 'input_size'
        even after 'create_objective_function' has finished execution.
        as they are captured by the closure.
        """
        # Dynamically suggest hyperparameters based on model_info
        hyperparams = {}
        for param_name, param_config in model_info["hyperparameters"].items():
            param_type = param_config["type"]
            match param_type:
                case "int":
                    hyperparams[param_name] = trial.suggest_int(
                        param_name, param_config["low"], param_config["high"]
                    )
                case "float":
                    hyperparams[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=param_config.get("log", default=False),
                    )
                case "categorical":
                    hyperparams[param_name] = trial.suggest_categorical(
                        param_name, param_config["choices"]
                    )
                case _:
                    raise ValueError(f"Unknown parameter type: {param_type}")

        # Perform cross-validation and return the mean validation loss of the inner loop as objective
        val_losses = cross_validate(
            X, y, model_name, input_size, K_FOLDS, **hyperparams
        )
        return np.mean(val_losses)

    return objective


def main():
    X, y = load_dataset()
    # Split the data into training+validation set and test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_value = float("inf")
    best_params = None
    best_model_name = None

    for model_name, model_info in models_dict.items():
        logging.info(f"Training {model_name}...")

        input_size = get_input_size(model_name, X_train_val)

        objective_function = create_objective_function(
            X_train_val, y_train_val, model_name, input_size, model_info
        )
        study = optuna.create_study(
            direction="minimize",
            storage="sqlite:///db.sqlite3",
            study_name=f"{model_name}_{time_stamp()}",
        )
        study.optimize(objective_function, n_trials=3)

        if study.best_value < best_value:
            best_value = study.best_value
            best_params = study.best_params
            best_model_name = model_name

        logging.info(
            f"Best value for {model_name}: {study.best_value} (params: {study.best_params})"
        )

    logging.info(
        f"Overall Best Model: {best_model_name} with value: {best_value} (params: {best_params})"
    )

    # Retrain the model with the best parameters on the entire training+validation set
    train_loader, test_loader = create_dataloaders(
        X_train_val, y_train_val, X_test, y_test, batch_size=best_params["batch_size"]
    )
    input_size = get_input_size(best_model_name, X_train_val)
    model, criterion, optimizer = initialize_model(
        best_model_name, input_size, **best_params
    )
    train_model(
        model, train_loader, test_loader, criterion, optimizer, EPOCHS, is_test=True
    )
    average_loss, accuracy = evaluate_model(model, test_loader, criterion)
    logging.info(
        f"Final Model | Test Accuracy: {accuracy:.2f} | Test Loss: {average_loss:.2f}"
    )


if __name__ == "__main__":
    main()


"""

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    scheduler_type = trial.suggest_categorical("scheduler_type", ["StepLR", "ExponentialLR", "CosineAnnealingLR"])
    
    if scheduler_type == "StepLR":
        step_size = trial.suggest_int("step_size", 1, 30)
        gamma = trial.suggest_float("gamma", 0.1, 1.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "ExponentialLR":
        gamma = trial.suggest_float("gamma", 0.1, 1.0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_type == "CosineAnnealingLR":
        T_max = trial.suggest_int("T_max", 1, 100)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    
    # Training loop here, including calls to scheduler.step() as appropriate
    
    return val_loss  # Return a metric to minimize or maximize




import optuna
import torch
import torch.nn as nn
import torch.optim as optim

# Assume some model, dataset, and DataLoader setup
model = ... # Your model
train_loader = ... # Your training DataLoader
val_loader = ... # Your validation DataLoader

def objective(trial):
    # Hyperparameters to optimize, including learning rate scheduler parameters
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    scheduler_step_size = trial.suggest_int("scheduler_step_size", 1, 30)
    scheduler_gamma = trial.suggest_float("scheduler_gamma", 0.1, 1.0)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    
    for epoch in range(num_epochs):
        for batch in train_loader:
            # Training loop
            optimizer.zero_grad()
            output = model(batch)
            loss = ... # Compute loss
            loss.backward()
            optimizer.step()
        scheduler.step()  # Update the learning rate
        
        # Validate your model
        val_loss = ... # Compute validation loss
    
    return val_loss  # Objective to minimize

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)


"""
