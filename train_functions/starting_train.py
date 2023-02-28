import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np



def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval, label_dict):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )


    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()


    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):
            # TODO: Backpropagation and gradient descent
            y_hat = model(batch[0])
            labels = batch[1]
            y = torch.zeros(batch_size, dtype=torch.long)
            for i, s in enumerate(labels):

                y[i] = label_dict[s]




            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.

                pred = np.zeros(y_hat.shape[0])
                for i in range(y_hat.shape[0]):
                    pred[i] = torch.argmax(y_hat[i]) + 1

                accuracy = compute_accuracy(pred, y)
                print("Accuracy after {0} step: {1}".format(step, accuracy))


                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!

                # evaluate(val_loader, model, loss_fn)

            step += 1

        print()


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

    n_correct = 0

    for i in range(len(outputs)):
        if outputs[i]==labels[i]:
            n_correct += 1
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    """
    y_hat = model(val_loader)
    loss = loss_fn(y_hat, [row[1] for row in val_loader])
    return loss, compute_accuracy(y_hat, [row[1] for row in val_loader])
