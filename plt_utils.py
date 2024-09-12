import matplotlib.pyplot as plt


def save_imshow(data, title, path, scatter_kwargs, **plt_kwargs):
    scatter_kwargs = {**scatter_kwargs}

    fig = plt.figure()
    plt.imshow(data, **plt_kwargs)
    plt.colorbar()

    # Plot receptors
    if scatter_kwargs.get('number_of_receptors') is not None:
        for i in range(scatter_kwargs['number_of_receptors']):
            plt.scatter(scatter_kwargs['receptor_x'][i], scatter_kwargs['receptor_z'][i], s=1, color='green')

    plt.title(title)
    plt.grid()
    plt.savefig(path)
    plt.close(fig)


def plot_imshow(data, title, scatter_kwargs, **plt_kwargs):
    scatter_kwargs = {**scatter_kwargs}

    plt.figure()
    plt.imshow(data, **plt_kwargs)
    plt.colorbar()

    # Plot receptors
    if scatter_kwargs.get('number_of_receptors') is not None:
        for i in range(scatter_kwargs['number_of_receptors']):
            plt.scatter(scatter_kwargs['receptor_x'][i], scatter_kwargs['receptor_z'][i], s=1, color='green')

    plt.title(title)
    plt.grid()
    plt.show()
