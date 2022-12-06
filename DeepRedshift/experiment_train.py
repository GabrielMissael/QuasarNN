
import wandb
from torch.utils.data import DataLoader
from torch import nn
import torch
from DeepRedshift.train_eval import train, get_predictions
from DeepRedshift.figures import report_plot
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def model_try(config, model, train_set, val_set):

    wandb.init(project='QuasarNN', entity = 'gmissaelbarco', config=config)

    epochs = config['epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    dropout = config['dropout']
    model_type = config['model_type']
    layers_dims = config['layers_dims']

    # Create dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # Create optimize and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.L1Loss()

    # Send model to device
    model.to(device)

    # Train model
    model.train()
    train_losses, val_losses = train(model, train_loader, val_loader, loss_fn, optimizer, epochs)

    # Get predictions
    model.eval()
    quasars, labels, predictions, metrics_values = get_predictions(model, val_loader)

    # Send metrics to wandb
    wandb.run.summary['mae'] = metrics_values['mae']
    wandb.run.summary['mse'] = metrics_values['mse']
    wandb.run.summary['ccc'] = metrics_values['ccc']
    wandb.run.summary['r2'] = metrics_values['r2']

    # Save model architecture in wandb
    wandb.save('model.pt')

    # Report plot
    report = report_plot(labels, predictions, train_losses, val_losses, config, metrics_values)

    # Send report to wandb
    wandb.log({"Report": wandb.Image(report)})

    # Save report figure
    # If model_type directory does not exist, create it
    if not os.path.exists('reports/'+model_type):
        os.makedirs('reports/'+model_type)
        max_index = 1
    else:
        max_index = max([int(file.split('.')[0]) for file in os.listdir('reports/'+model_type)]) + 1

    # Save figure
    report.savefig(os.path.join('reports/'+ model_type, str(max_index) + '.png'))

    # Save file name in wandb
    wandb.run.summary['report_path'] = str(os.path.join(model_type, str(max_index) + '.png'))

    # Finish run
    wandb.finish()

    # Clear GPU memory
    torch.cuda.empty_cache()

    del model, optimizer, loss_fn, train_loader, val_loader,
    del quasars, metrics_values, labels, predictions, train_losses, val_losses
