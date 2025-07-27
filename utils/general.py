import logging

import numpy as np
import torch
from colorama import Fore, Style

def create_bar_chart_image(values, labels=None, title="Bar Chart"):
    """
    Create a bar chart and return it as a TensorFlow image tensor.
    Use this if you want to handle the TensorBoard logging yourself.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(values))

    if labels is None:
        labels = [f"Item {i + 1}" for i in range(len(values))]

    bars = ax.bar(x_pos, values, alpha=0.8, color='steelblue')

    ax.set_xlabel('Categories')
    ax.set_ylabel('Values')
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + max(values) * 0.01,
                f'{value:.2f}', ha='center', va='bottom')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    plt.close(fig)
    buf.close()

    return image
