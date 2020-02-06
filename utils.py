import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os


def compute_stats_ar(results, ar_params, verbose=False):
    weights = results["weights"]
    error = results["predicted"] - results["actual"]
    stats = {}

    abs_error = np.abs(weights - ar_params)

    symmetric_abs_coeff = np.abs(weights) + np.abs(ar_params)
    stats["sMAPE (AR-coefficients)"] = 100 * np.mean(abs_error / (10e-9 + symmetric_abs_coeff))

    sTPE = 100 * np.sum(abs_error) / (10e-9 + np.sum(symmetric_abs_coeff))
    stats["sTPE (AR-coefficients)"] = sTPE

    # abs_error_sum = sum(abs_error)
    # stats["TP (AR-coefficients)"] = min(1.0, abs_error_sum / (sum(np.abs(data["ar"])) + 10e-9))
    # mean_rel_error = np.mean(np.minimum(1.0, abs_error / (np.abs(data["ar"]) + 10e-9)))
    # stats["mean_rel_error"] = mean_rel_error

    # predictions error
    stats["MSE"] = np.mean(error ** 2)

    if verbose:
        print("MSE: {}".format(stats["MSE"]))
        print("sMAPE (AR-coefficients): {:6.3f}".format(stats["sMAPE (AR-coefficients)"]))
        print("sTPE (AR-coefficients): {:6.3f}".format(stats["sTPE (AR-coefficients)"]))
        # print("Relative error: {:6.3f}".format(stats["TP (AR-coefficients)"]))
        # print("Mean relative error: {:6.3f}".format(mean_rel_error))

        print("AR params: ")
        print(ar_params)

        print("Weights: ")
        print(weights)
    return stats


def plot_loss_curve(losses, test_loss=None, epoch_losses=None, show=False, save=False):
    fig = plt.figure()
    fig.set_size_inches(12, 6)
    ax = plt.axes()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    x_loss = list(range(len(losses)))
    plt.plot(x_loss, losses, 'b', alpha=0.3)
    if epoch_losses is not None:
        iter_per_epoch = int(len(losses) / len(epoch_losses))
        epoch_ends = int(iter_per_epoch/2) + iter_per_epoch*np.arange(len(epoch_losses))
        plt.plot(epoch_ends, epoch_losses, 'b')
    if test_loss is not None:
        plt.hlines(test_loss, xmin=x_loss[0], xmax=x_loss[-1])
    if save:
        if not os.path.exists('results'):
            os.makedirs('results')
        figname = 'results/loss_DAR.png'
        plt.savefig(figname, dpi=600, bbox_inches='tight')
    plt.show()
    # plt.close()


def plot_prediction_sample(predicted, actual, num_obs=100, model_name="AR-Net", save=False):
    fig2 = plt.figure()
    fig2.set_size_inches(10, 6)
    plt.plot(actual[0:num_obs])
    plt.plot(predicted[0:num_obs])
    plt.legend(["Actual Time-Series", "{}-Prediction".format(model_name)])
    if save:
        if not os.path.exists('results'):
            os.makedirs('results')
        figname = 'results/prediction_{}.png'.format(model_name)
        plt.savefig(figname, dpi=600, bbox_inches='tight')
    plt.show()


def plot_error_scatter(predicted, actual, model_name="AR-Net", save=False):
    # error = predicted - actual
    fig3 = plt.figure()
    fig3.set_size_inches(6, 6)
    plt.scatter(actual, predicted - actual, marker='o', s=10, alpha=0.3)
    plt.legend(["{}-Error".format(model_name)])
    if save:
        if not os.path.exists('results'):
            os.makedirs('results')
        figname = 'results/scatter_{}.png'.format(model_name)
        plt.savefig(figname, dpi=600, bbox_inches='tight')
    plt.show()


def plot_weights(ar_val, weights, ar, model_name="AR-Net", save=False):
    df = pd.DataFrame(
        zip(
            list(range(1, ar_val + 1)) * 2,
            ["AR-Process (True)"] * ar_val + [model_name] * ar_val,
            list(ar) + list(weights)
        ),
        columns=["AR-coefficient (lag number)", "model", "value (weight)"]
    )
    plt.figure(figsize=(10, 6))
    palette = {"Classic-AR": "C0", "AR-Net": "C1", "AR-Process (True)": "k"}
    sns.barplot(x="AR-coefficient (lag number)", hue="model", y="value (weight)", data=df)
    if save:
        if not os.path.exists('results'):
            os.makedirs('results')
        figname = 'results/weights_{}_{}.png'.format(ar_val, model_name, palette=palette)
        plt.savefig(figname, dpi=600, bbox_inches='tight')

    plt.show()


def plot_results(results, model_name="MODEL", save=False):
    plot_prediction_sample(results["predicted"], results["actual"], num_obs=100, model_name=model_name, save=save)
    plot_error_scatter(results["predicted"], results["actual"], model_name=model_name, save=save)


def jsonize(results):
    for key, value in results.items():
        if type(value) is list:
            if type(value[0]) is list:
                results[key] = [["{:8.5f}".format(xy) for xy in x] for x in value]
            else:
                results[key] = ["{:8.5f}".format(x) for x in value]
        else:
            results[key] = "{:8.5f}".format(value)
    return results


def list_of_dicts_2_dict_of_lists(sources):
    keys = sources[0].keys()
    res = {}
    for key in keys:
        res[key] = [d[key] for d in sources]
    return res


def list_of_dicts_2_dict_of_means(sources):
    keys = sources[0].keys()
    res = {}
    for key in keys:
        res[key] = np.mean([d[key] for d in sources])
    return res


def list_of_dicts_2_dict_of_means_minmax(sources):
    keys = sources[0].keys()
    res = {}
    for key in keys:
        values = [d[key] for d in sources]
        res[key] = (np.mean(values), min(values), max(values))
        # res["mean"][key] = np.mean(values)
        # res["min"][key] = min(values)
        # res["max"][key] = max(values)
    return res


def get_json_filenames(values, subdir=None):
    ar_filename = get_json_filenames_type("AR", values, subdir)
    dar_filename = get_json_filenames_type("DAR", values, subdir)
    return ar_filename, dar_filename


def get_json_filenames_type(model_type, values, subdir=None):
    filename = 'results/{}{}_{}.json'.format(
        subdir + "/" if subdir is not None else "",
        model_type,
        "-".join([str(x) for x in values])
    )
    return filename


def intelligent_regularization(sparsity):
    if sparsity is not None:
        # best:
        # lam = 0.01 * (1.0 / sparsity - 1.0)
        lam = 0.02 * (1.0 / sparsity - 1.0)
        # lam = 0.05 * (1.0 / sparsity - 1.0)

        # alternatives
        # l1 = 0.02 * (np.log(2) / np.log(1 + sparsity) - 1.0)
        # l1 = 0.1 * (1.0 / np.sqrt(sparsity) - 1.0)
    else:
        lam = 0.0
    return lam


