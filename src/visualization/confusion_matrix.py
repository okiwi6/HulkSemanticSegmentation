from ..configuration.constants import CLASS_TO_LABEL, NUM_CLASSES

import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.classification import MulticlassConfusionMatrix
import torch

def fig_confusion_matrix(predictions, labels):
    fig = plt.figure()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mccm = MulticlassConfusionMatrix(num_classes=NUM_CLASSES, normalize="true").to(device)
    mccm.update(predictions, labels)
    values = mccm.compute().cpu().numpy()
    
    ax = sns.heatmap(values, annot=True, fmt='.2%', xticklabels=CLASS_TO_LABEL, yticklabels=CLASS_TO_LABEL)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    fig.add_axes(ax)
    return fig