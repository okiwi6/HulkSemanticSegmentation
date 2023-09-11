from ..configuration.constants import CLASS_TO_LABEL

import numpy as np
import matplotlib.pyplot as plt

def fig_histogram(histograms: list[np.array], titles: list[str]):
    fig, axes = plt.subplots(1, len(histograms))

    for i, (hist, title) in enumerate(zip(histograms, titles)):
        axes[i].bar(CLASS_TO_LABEL, hist, label=CLASS_TO_LABEL)
        axes[i].set_title(title)

    return fig