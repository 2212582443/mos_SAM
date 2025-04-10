import matplotlib.pyplot as plt


def plot_segment(imgae, segment, alpha=0.4):
    """Plot segment on image
    Args:
        imgae: [h, w]
        segment: [h, w]
    """
    plt.imshow(imgae, cmap="gray")
    if segment is not None:
        plt.imshow(segment, cmap="jet", alpha=alpha)
    plt.show()


import matplotlib.pyplot as plt
import numpy as np


def grid2contour(grid, title):
    """
    grid--image_grid used to show deform field
    type: numpy ndarray, shape： (h, w, 2), value range：(-1, 1)
    """
    assert grid.ndim == 3
    x = np.arange(-1, 1, 2.0 / grid.shape[1])
    y = np.arange(-1, 1, 2.0 / grid.shape[0])
    X, Y = np.meshgrid(x, y)
    Z1 = grid[:, :, 0] + 2  # remove the dashed line
    Z1 = Z1[::-1]  # vertical flip
    Z2 = grid[:, :, 1] + 2

    plt.figure()
    plt.contour(X, Y, Z1, 15, levels=50, colors="k")  # 改变levels的值，可以改变形变场的密集程度
    plt.contour(X, Y, Z2, 15, levels=50, colors="k")
    plt.xticks(()), plt.yticks(())  # remove x, y ticks
    plt.title(title)
    plt.show()
    plt.savefig
