import matplotlib.pyplot as plt
import numpy as np

def plot_FN_FP_RT_shadow_price(FP, FN, range=[-10, 10, -10000,200000]):

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create a heatmap of the confusion matrix
    # ax.plot(np.log10(FP.cumsum()), color="red")
    # ax.plot(np.log10(FN.cumsum()), color="Black")
    ax.plot(FP.cumsum(), color="red")
    ax.plot(FN.cumsum(), color="Black")

    # add to tge image the sum of FP, as a text at the end of the curve
    ax.text(
        len(FP) - 1,
        FP.cumsum()[-1],
        f"${FP.cumsum()[-1]:,.0f}",
        ha="center",
        va="bottom",
        color="red",
    )
    ax.text(
        len(FN) - 1,
        FN.cumsum()[-1],
        f"${FN.cumsum()[-1]:,.0f}",
        ha="center",
        va="bottom",
        color="black",
    )
    ax.set_xlabel("Daily Event (from greatest FORECAST RT shadow price to smallest)")
    ax.set_ylabel("Accumulated Shadow Price")
    ax.legend(["False Positive", "False Negative"])
    ax.set_title("Accumulated Shadow Price of False Positive and False Negative")

    # # Set the title and labels for the plot
    # ax.set_title("Confusion Matrix", fontsize=16)
    # ax.set_ylabel('Predicted', fontsize=14)

    # zoom in the image in the area of interest which is the first 20 days, and putting the zoomed image in the middle of the image
    axins = ax.inset_axes([0.4, 0.3, 0.4, 0.4])  # zoom = 6
    axins.plot(FP.cumsum()[:120], color="red")
    axins.plot(FN.cumsum()[:120], color="Black")
    axins.set_xlim(range[0], range[1])
    axins.set_ylim(range[2], range[3])
    axins.set_xlabel('X-axis label')  # Add x-axis label
    axins.set_ylabel('Y-axis label')  # Add y-axis label
    axins.set_title('Area of Interest')
    ax.indicate_inset_zoom(axins)

    # fig.tight_layout(
    #     rect=[0, 0.03, 1, 0.95]
    # )  # Adjust the top spacing to accommodate the "Actual" label
