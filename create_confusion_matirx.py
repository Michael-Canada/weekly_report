import matplotlib.pyplot as plt
import numpy as np

# Define the data
true_positive = 0.54
false_positive = 0.26
false_negative = 0.20

def plot_confusion_matrix(true_positive, false_positive, false_negative):



    true_negative = 1 - true_positive - false_positive - false_negative

    # Create the confusion matrix
    confusion_matrix = np.array([[true_positive, false_negative],
                                [false_positive, true_negative]])

    # Define the labels for the matrix
    labels = ['Positive', 'Negative']

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create a heatmap of the confusion matrix
    im = ax.imshow(confusion_matrix, cmap='Blues')

    # # Add colorbar
    # cbar = ax.figure.colorbar(im, ax=ax)
    # cbar.ax.set_ylabel('Percentage', rotation=-90, va="bottom")

    # Set the ticks and labels for the x and y axes
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add labels to each cell of the matrix
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(i, j, f'{confusion_matrix[j, i]:.0%}',
                    ha="center", va="center", color="white" if confusion_matrix[j, i] > 0.5 else "black")

    # Set the title and labels for the plot
    # ax.set_title("Confusion Matrix", fontsize=16)
    ax.set_ylabel('Predicted', fontsize=14)

    # Add the "Actual" label above the chart
    ax.text(0.5, 1.1, 'Actual', ha='center', va='center', transform=ax.transAxes, fontsize=14)

    # Adjust the layout and display the plot
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the top spacing to accommodate the "Actual" label
    # plt.show()
