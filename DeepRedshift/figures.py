import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

sns.set_theme()

def scatter_plot_eval(labels, predictions):
    fig = plt.figure()

    mean = np.mean(np.abs(labels - predictions))
    std = np.std(np.abs(labels - predictions))

    plt.scatter(labels, predictions, s=10, alpha = 0.2, color = 'green')
    plt.plot([0, 6], [0, 6], color='black', linestyle='--', zorder=10)
    plt.xlim(min(labels)- 0.1, max(labels) + 0.1)
    plt.ylim(min(labels)-0.1, max(labels)+0.1)
    plt.title('Predicted vs. actual redshift\n Mean absolute error: {:.5f} \u00B1 {:.5f}'.format(mean, std))
    plt.xlabel('Redshift')
    plt.ylabel('Predicted redshift')
    return fig

def error_plot_eval(labels, predictions):
    idx_sort = np.argsort(labels)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ax[0].plot(predictions[idx_sort], label = 'Prediction', linewidth=0.3)
    ax[0].plot(labels[idx_sort], label = 'Real', linewidth=2, color = 'black')
    ax[0].set_ylabel('$Z$')
    ax[0].set_ylim(1.5, max(labels[idx_sort]) +1)
    ax[0].legend()

    ax[1].plot(labels[idx_sort] - predictions[idx_sort], linewidth=0.3, color = 'black')
    ax[1].set_ylabel('$\Delta Z$')
    ax[1].set_xlabel('Quasar ID')
    ax[1].set_ylim(-1, 1)
    return fig

def loss_eval(train_losses, val_losses):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[1].plot(val_losses, label='Validation loss')
    ax[1].plot(train_losses, label='Train loss')
    ax[1].set_yscale('log')
    ax[1].legend()
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')

    ax[0].plot(val_losses, label='Validation loss')
    ax[0].plot(train_losses, label='Train loss')
    ax[0].legend()
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    return fig

def report_plot(labels, predictions, train_losses, val_losses, config, metrics_values):
    sns.set_theme()
    fig = plt.figure(constrained_layout = True, figsize=(8, 11))

    subfigs = fig.subfigures(4, 1, wspace=0.05, hspace=0.01, width_ratios=[1], height_ratios=[0.6, 1.5, 1, 1.5])

    subfigs[0].suptitle(f'Model details, hyperparemeters and metrics', fontsize=20)
    subfigs[1].suptitle('Model performance', fontsize=16)
    subfigs[2].suptitle('Error distribution', fontsize=16)
    subfigs[3].suptitle('Losses', fontsize=16)

    # write model details
    ax_details = subfigs[0].subplots(1, 1)
    ax_details.axis('off')
    t = (
        f'Model type = {config["model_type"]} - {config["layers_dims"]}\n' + \
        f'Learning rate = {config["learning_rate"]}, epochs = {config["epochs"]}, ' + \
        f'Batch size = {config["batch_size"]}, Dropout = {config["dropout"]}\n' + \
        f'MAE = {metrics_values["mae"]:.6f}, MSE = {metrics_values["mse"]:.6f}, ' + \
        f'CCC = {metrics_values["ccc"]:.6f}, R2 = {metrics_values["r2"]:.6f}'
    ).expandtabs()

    ax_details.text(0.5, 0.5, t, fontsize=13, verticalalignment='center',
                    horizontalalignment='center', wrap = True,
                    bbox=dict(facecolor='#EAEAF2', boxstyle='round', pad=1))

    # Plot performance
    ax_perf = subfigs[1].subplots(1, 2)

    mean = np.mean(np.abs(labels - predictions))
    std = np.std(np.abs(labels - predictions))

    ax_perf[0].scatter(labels, predictions, s=10, alpha = 0.2, color = 'green')
    ax_perf[0].plot([-1, 6], [-1, 6], color='black', linestyle='--', zorder=10)
    ax_perf[0].set_xlim(min(labels)- 0.1, max(labels) + 0.1)
    ax_perf[0].set_ylim(min(labels)-0.1, max(labels)+0.1)
    ax_perf[0].set_title('Predicted vs. actual redshift')
    ax_perf[0].set_xlabel('Real redshift')
    ax_perf[0].set_ylabel('Predicted redshift')

    idx_sort = np.argsort(labels)

    #ax_perf[1].plot(predictions[idx_sort], label = 'Prediction', linewidth=0.3)
    #ax_perf[1].plot(labels[idx_sort], label = 'Real', linewidth=2, color = 'black')
    #ax_perf[1].set_ylabel('$Z$')
    #ax_perf[1].set_ylim(1.5, max(labels[idx_sort]) +1)
    #ax_perf[1].legend()
    #ax_perf[1].set_xlabel('Quasar index')
    #ax_perf[1].set_title('Redshift error for increasing values')

    delta_vel = (labels - predictions)/(1+labels)*300_000
    delta_vel_mean = np.mean(delta_vel)
    delta_vel_std = np.std(delta_vel)

    # 50 bins with 3 std
    bins = np.linspace(-2.5*delta_vel_std, 2.5*delta_vel_std, 50)

    hist = ax_perf[1].hist(delta_vel, bins = bins,
        color = 'blue', alpha = 0.5, label = 'This', density = True,
        histtype = 'stepfilled')
    # Set ylimit with highest bin
    ax_perf[1].set_ylim(0, max(hist[0]))

    # Set xlim to 3 std
    ax_perf[1].set_xlim(-2.5*delta_vel_std, 2.5*delta_vel_std)

    # Plot error distribution from quasarNet
    # mean = 8, std = 664
    aux = np.linspace(-3000,  3000, 1000)
    ax_perf[1].plot(aux, stats.norm.pdf(aux, 8, 664),
        color = 'black', label = 'QuasarNet', linestyle = '--', linewidth = 2)

    ax_perf[1].set_xlabel('$\Delta v$ [km/s]')
    ax_perf[1].set_ylabel('Density')
    ax_perf[1].set_title(f'$\Delta v$ = {delta_vel_mean:.2f} $\pm$ {delta_vel_std:.2f} km/s')
    ax_perf[1].legend()

    # Plot error distribution
    ax_err = subfigs[2].subplots(1, 1)

    ax_err.plot(labels[idx_sort] - predictions[idx_sort], linewidth=0.3, color = 'black')
    ax_err.set_ylabel('$\Delta Z$')
    ax_err.set_xlabel('Quasar index')
    ax_err.set_ylim(-1, 1)

    # Plot losses
    ax_loss = subfigs[3].subplots(1, 2)
    ax_loss[1].plot(val_losses, label='Validation loss')
    ax_loss[1].plot(train_losses, label='Train loss')
    ax_loss[1].set_yscale('log')
    ax_loss[1].legend()
    ax_loss[1].set_xlabel('Epoch')
    ax_loss[1].set_ylabel('Loss')

    ax_loss[0].plot(val_losses, label='Validation loss')
    ax_loss[0].plot(train_losses, label='Train loss')
    ax_loss[0].legend()
    ax_loss[0].set_xlabel('Epoch')
    ax_loss[0].set_ylabel('Loss')
    return fig
