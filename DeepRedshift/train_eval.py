
from tqdm import tqdm
import torch
import numpy as np
import wandb
from torchmetrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    ConcordanceCorrCoef,
    PearsonCorrCoef,
    R2Score)
from DeepRedshift.figures import report_plot

# Define metrics
#MeanAbsoluteError, MeanSquaredError, ConcordanceCorrCoef, PearsonCorrCoef, R2Score
metrics = {
    'mae': MeanAbsoluteError(),
    'mse': MeanSquaredError(),
    'ccc': ConcordanceCorrCoef(num_outputs=1),
    'r2': R2Score()
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train(model, train_loader, val_loader, loss_fn, optimizer, epochs):
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs)):
        train_loss = 0
        val_loss = 0
        model.train()
        for batch in train_loader:
            # Get data
            quasar, label = batch
            quasar = quasar.type(torch.Tensor).to(device)
            label = label.type(torch.Tensor).to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            pred = model(quasar)
            loss = loss_fn(pred.flatten(), label)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update loss
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                # Get data
                quasar, label = batch
                quasar = quasar.type(torch.Tensor).to(device)
                label = label.type(torch.Tensor).to(device)

                # Forward pass
                pred = model(quasar)
                loss = loss_fn(pred.flatten(), label)

                # Update loss
                val_loss += loss.item()

        # Update losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Log metrics
        wandb.log({'train_loss': train_loss, 'val_loss': val_loss})

    return train_losses, val_losses

# Get predictions
def get_predictions(model, loader):
    model.eval()
    quasars = []
    labels = []
    predictions = []
    metrics_values = {}
    with torch.no_grad():
        for batch in loader:
            # Get data
            quasar, label = batch
            quasar = quasar.type(torch.Tensor).to(device)
            label = label.type(torch.Tensor).to(device)

            # Forward pass
            pred = model(quasar)

            # Update lists
            for i in range(len(quasar)):
                quasars.append(quasar[i].cpu().numpy())
                labels.append(float(label[i].cpu().numpy()))
                predictions.append(float(pred[i].cpu().numpy()))
    labels = np.array(labels)
    predictions = np.array(predictions)
    metrics_values = {
        'mae': metrics['mae'](torch.from_numpy(labels), torch.from_numpy(predictions)),
        'mse': metrics['mse'](torch.from_numpy(labels), torch.from_numpy(predictions)),
        'ccc': metrics['ccc'](torch.from_numpy(labels), torch.from_numpy(predictions)),
        'r2': metrics['r2'](torch.from_numpy(labels), torch.from_numpy(predictions))
    }
    return quasars, labels, predictions, metrics_values

