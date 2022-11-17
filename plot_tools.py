import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def simple_plot(
    Y, X=[], title="", xlabel="", ylabel="", show=False, save_path="", x_begin=1
):
    plt.cla()
    plt.clf()
    if X == []:
        X = [i + x_begin for i in range(len(Y))]
    plt.plot(X, Y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if save_path != "":
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()

def complex_plot(
    my_array,
    mode="horizontal",
    title="",
    xlabel="",
    ylabel="",
    show=False,
    save_path="",
    x_begin=1,
    legend=[],
    y_lim=(0, 0),
    x_lim=(0, 0),
):
    plt.cla()
    plt.clf()
    if mode == "vertical":
        X = [i + x_begin for i in range(len(my_array))]
        plt.legend([str(i + 1) for i in range(len(my_array))])
        for j in range(len(my_array[0])):
            Y = [my_array[i][j] for i in range(len(my_array))]
            plt.plot(X, Y)
    elif mode == "horizontal":
        plt.legend([str(i + 1) for i in range(len(my_array[0]))])
        for i in range(len(my_array)):
            X = [i + x_begin for i in range(len(my_array[i]))]
            Y = [my_array[i][j] for j in range(len(my_array[i]))]
            plt.plot(X, Y)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if x_lim != (0, 0):
        plt.xlim(x_lim)
    if y_lim != (0, 0):
        plt.ylim(y_lim)
    if legend != []:
        plt.legend(legend)
    if save_path != "":
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()

def plot_3d(Z, title="", xlabel="", ylabel="", show=False, save_path="", z_lim=[0, 0]):
    plt.cla()
    shape = Z.shape
    ax = plt.axes(projection="3d")
    x = [i for i in range(shape[0])]
    y = [i for i in range(shape[1])]
    X, Y = np.meshgrid(x, y)
    Z = np.transpose(Z)
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if z_lim != [0, 0]:
        ax.set_zlim(z_lim)
    if save_path != "":
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
