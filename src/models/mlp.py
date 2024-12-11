import logging
from datetime import datetime

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.log_config import configure_logging

configure_logging()

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)
logging.debug(f"Using device: {DEVICE}")

EPOCHS = 10
LEARNING_RATE = 0.001  # not used
BATCH_SIZE = 32  # not used
K_FOLDS = 5


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


def initialize_model(input_size, hidden_size, learning_rate):
    model = MLP(input_size, hidden_size).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer


def load_dataset(X_to_2d=False):
    X = np.load("data/data.npy")
    y = np.load("data/target.npy")
    if X_to_2d:
        X = reshape_features_to_2d(X)
    return X, y


def reshape_features_to_2d(X):
    if len(X.shape) != 3:
        raise ValueError("Input array should be a 3D array.")
    return np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]), order="F")


def create_dataloaders(
    X_train, y_train, X_test, y_test, batch_size, is_validation=False
):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)  # use the same scaler as for training data

    train_data = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1)
    )
    test_data = TensorDataset(
        torch.FloatTensor(X_test), torch.FloatTensor(y_test).view(-1, 1)
    )

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

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
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
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
            f"{phase} | Epoch {epoch+1}/{num_epochs}, "
            f"Loss: {epoch_loss:.4f}, {data_set} "
            f"Loss: {val_loss:.4f}, {data_set} "
            f"Accuracy: {val_accuracy:.4f}"
        )


def evaluate_model(
    model,
    val_loader,
    criterion,
):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            y_pred_logits = model(X_batch)
            loss = criterion(y_pred_logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)

            y_pred_classes = (torch.sigmoid(y_pred_logits) >= 0.5).float()
            total += y_batch.size(0)
            correct += (y_pred_classes == y_batch).sum().item()

    average_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return average_loss, accuracy


def cross_validate(
    X,
    y,
    input_size,
    hidden_size,
    k_folds,
    lr,
    batch_size,
):
    kfold = KFold(n_splits=k_folds, shuffle=True)
    val_losses = []

    for fold, (train_index, val_index) in enumerate(kfold.split(X, y)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        train_loader, val_loader = create_dataloaders(
            X_train, y_train, X_val, y_val, batch_size=batch_size, is_validation=True
        )

        model, criterion, optimizer = initialize_model(
            input_size=input_size, hidden_size=hidden_size, learning_rate=lr
        )

        train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)
        average_loss, accuracy = evaluate_model(model, val_loader, criterion)
        val_losses.append(average_loss)
        logging.info(
            f"Fold {fold} | Accuracy: {accuracy:.2f} | Validation Loss: {average_loss:.2f},"
        )
    return val_losses


def create_objective_function(X, y, input_size):
    def objective(trial):
        """
        The 'objective' function can access 'X', 'y', and 'input_size'
        even after 'create_objective_function' has finished execution.
        as they are captured by the closure.
        """
        # Suggest values for the hyperparameters
        hidden_size = trial.suggest_int("hidden_size", 128, 1024)
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

        # Perform cross-validation and return the mean validation loss of the inner loop as objective
        val_losses = cross_validate(
            X, y, input_size, hidden_size, K_FOLDS, lr, batch_size
        )
        return np.mean(val_losses)

    return objective


def main():
    X, y = load_dataset(X_to_2d=True)
    # Split the data into training+validation set and test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    input_size = X_train_val.shape[1]

    objective_function = create_objective_function(X_train_val, y_train_val, input_size)

    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///db.sqlite3",
        study_name=f"clips_{time_stamp()}",
    )
    study.optimize(objective_function, n_trials=100)

    best_params = study.best_trial.params
    logging.info(f"Best value: {study.best_value} (params: {study.best_params})")

    # Retrain the model with best parameters on the entire training+validation set
    train_loader, test_loader = create_dataloaders(
        X_train_val, y_train_val, X_test, y_test, batch_size=best_params["batch_size"]
    )

    model, criterion, optimizer = initialize_model(
        input_size=input_size,
        hidden_size=best_params["hidden_size"],
        learning_rate=best_params["lr"],
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
