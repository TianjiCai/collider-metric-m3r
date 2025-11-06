import nn_utils
import numpy as np
import ot_utils
import comparison_utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch


def load_data(data_path, file_names):
    datasets = []
    for file_name in file_names:
        data = np.load(data_path + file_name)
        datasets.append(data)
    return datasets


def main():
    device = "mps"
    n_classes = 8
    anchors_per_class = 100
    events_per_class = 100000

    rel_reps = np.load("rel_rep.npy")
    labels = np.repeat(np.arange(n_classes), events_per_class)[1:]

    X_train, X_test, y_train, y_test = train_test_split(
        rel_reps, labels, test_size=0.2, stratify=labels, random_state=42
    )

    model = nn_utils.MLP(
        input_dim=n_classes * anchors_per_class,
        hidden_dims=[2048, 1024, 256, 128],
        output_dim=n_classes,
    )

    # model = nn_utils.ResNet(
    #     input_dim=n_classes * anchors_per_class,
    #     hidden_dim=1024,
    #     output_dim=n_classes,
    #     num_blocks=3,
    # )
    model = torch.load("trained_model.pth", map_location=device)
    print("Warm start training...")

    trained_model, train_loss_history, val_loss_history = nn_utils.train_network_torch(
        model,
        X_train,
        y_train,
        val_size=0.2,
        epochs=256,
        patience=None,
        batch_size=512,
        learning_rate=1e-4,
        device=device,
    )
    
    # save the trained model
    torch.save(trained_model.state_dict(), "trained_model.pth")

    print(nn_utils.evaluate_network_torch(trained_model, X_test, y_test, device=device))


if __name__ == "__main__":
    main()
