import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dims, output_dim, activation=F.relu, dropout=0.0
    ):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU() if activation == F.relu else nn.Tanh())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        out += residual
        out = self.activation(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=3, dropout=0.0):
        super(ResNet, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]

        for _ in range(num_blocks):
            layers.append(ResNetBlock(hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_network_torch(
    model,
    X_train,
    y_train,
    val_size=0.2,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=torch.optim.AdamW,
    optimizer_params={},
    epochs=100,
    patience=None,
    batch_size=256,
    learning_rate=1e-4,
    seed=42,
    device="cpu",
):
    set_seed(seed)
    model.train()

    if device == "cpu":
        device = torch.device("cpu")
        print("Using CPU for training.")
    elif device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using Nvidia GPU for training.")
    elif device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU for training.")
    else:
        raise ValueError(f"Unsupported device type: {device}")

    model.to(device)

    if val_size is None:
        val_size = 0.0

    optimizer = optimizer(model.parameters(), lr=learning_rate, **optimizer_params)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=seed
    )

    label_dtype = torch.long if len(np.unique(y_train)) > 2 else torch.float32

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=label_dtype)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=label_dtype)

    traindataset = torch.utils.data.TensorDataset(X_train, y_train)

    if val_size > 0:
        valdataset = torch.utils.data.TensorDataset(X_val, y_val)
    else:
        valdataset = None

    train_loader = torch.utils.data.DataLoader(
        traindataset, batch_size=batch_size, shuffle=True
    )

    if valdataset is not None:
        val_loader = torch.utils.data.DataLoader(
            valdataset, batch_size=batch_size, shuffle=False
        )
    else:
        val_loader = None

    train_loss_history = []
    val_loss_history = []
    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0
    num_samples = X_train.size(0)
    num_batches = (num_samples + batch_size - 1) // batch_size

    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss_history.append(epoch_loss / num_batches)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    loss = loss_fn(outputs, y)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            val_loss_history.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if patience is not None and epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label="Training Loss")
    if val_loader is not None:
        plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.xscale("log")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.show()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_loss_history, val_loss_history


def evaluate_network_torch(
    model, X_test, y_test, batch_size=256, device="cpu", seed=42
):
    set_seed(seed)

    if device == "cpu":
        device = torch.device("cpu")
        print("Using CPU for training.")
    elif device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using Nvidia GPU for training.")
    elif device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU for training.")
    else:
        raise ValueError(f"Unsupported device type: {device}")

    model.to(device)
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    num_samples = X_test_tensor.size(0)
    num_batches = (num_samples + batch_size - 1) // batch_size
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(num_batches):
            batch_X = X_test_tensor[i * batch_size : (i + 1) * batch_size]
            batch_y = y_test_tensor[i * batch_size : (i + 1) * batch_size]

            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy
