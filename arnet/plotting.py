import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_weights(ar_val, weights, ar=None, model_name="AR-Net", save=False, savedir="results"):
    if ar is not None:
        df = pd.DataFrame(
            zip(
                list(range(1, ar_val + 1)) * 2,
                ["AR-Process (True)"] * ar_val + [model_name] * ar_val,
                list(ar) + list(weights),
            ),
            columns=["AR-coefficient (lag number)", "model", "value (weight)"],
        )
    else:
        df = pd.DataFrame(
            zip(
                list(range(1, ar_val + 1)),
                [model_name] * ar_val,
                list(weights),
            ),
            columns=["AR-coefficient (lag number)", "model", "value (weight)"],
        )
    plt.figure(figsize=(10, 6))
    palette = {"Classic-AR": "C0", "AR-Net": "C1", "AR-Process (True)": "k"}
    sns.barplot(data=df, palette=palette, x="AR-coefficient (lag number)", hue="model", y="value (weight)")
    if save:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        figname = "weights_{}_{}.png".format(ar_val, model_name)
        plt.savefig(os.path.join(savedir, figname), dpi=600, bbox_inches="tight")
    else:
        plt.show()


def plot_prediction_sample(predicted, actual, num_obs=100, model_name="AR-Net", save=False, savedir="results"):
    fig2 = plt.figure()
    fig2.set_size_inches(10, 6)
    plt.plot(actual[0:num_obs])
    plt.plot(predicted[0:num_obs])
    plt.legend(["Actual Time-Series", "{}-Prediction".format(model_name)])
    if save:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        figname = "prediction_{}.png".format(model_name)
        plt.savefig(os.path.join(savedir, figname), dpi=600, bbox_inches="tight")
    else:
        plt.show()


def plot_error_scatter(predicted, actual, model_name="AR-Net", save=False, savedir="results"):
    # error = predicted - actual
    fig3 = plt.figure()
    fig3.set_size_inches(6, 6)
    plt.scatter(actual, predicted - actual, marker="o", s=10, alpha=0.3)
    plt.legend(["{}-Error".format(model_name)])
    if save:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        figname = "scatter_{}.png".format(model_name)
        plt.savefig(os.path.join(savedir, figname), dpi=600, bbox_inches="tight")
    else:
        plt.show()
