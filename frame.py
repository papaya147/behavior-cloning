import matplotlib.pyplot as plt
import math


def stack(obs, stack_size=4):
    unfolded = obs.unfold(0, stack_size, 1)
    stacked = unfolded.permute(0, 3, 1, 2)
    return stacked


def visualize(obs, mask=None, pred_mask=None, max_size=10):
    n2 = obs.shape[0]
    n = math.ceil(math.sqrt(n2))
    size = min(n * 5, max_size)
    plt.figure(figsize=(size, size))

    for i in range(n2):
        plt.subplot(n, n, i + 1)
        plt.imshow(obs[i], cmap="gray")
        if mask is not None:
            plt.imshow(mask[i], cmap="jet", alpha=0.5)
        if pred_mask is not None:
            plt.imshow(pred_mask[i], cmap="cool", alpha=0.4)
        plt.title(f"Frame {i + 1}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
