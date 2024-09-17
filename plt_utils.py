import matplotlib.pyplot as plt


def save_imshow(data, title, path, scatter_kwargs, **plt_kwargs):
    scatter_kwargs = {**scatter_kwargs}

    fig = plt.figure()
    plt.imshow(data, **plt_kwargs)
    plt.colorbar()

        ## # Plot reflectors
       # if scatter_kwargs.get('number_of_reflectors') is not None:
           # for i in range(scatter_kwargs['number_of_reflectors']):
               # plt.scatter(scatter_kwargs['reflector_x'][i], scatter_kwargs['reflector_z'][i], s=0.05, color='black')

        # Plot receptors
      #  if scatter_kwargs.get('number_of_receptors') is not None:
            #for i in range(scatter_kwargs['number_of_receptors']):
                #plt.scatter(scatter_kwargs['receptor_x'][i], scatter_kwargs['receptor_z'][i], s=0.05, color='white')

    plt.title(title)
    plt.grid()
    plt.savefig(path)
    plt.close(fig)


def save_imshow_4_subplots(nw_kwargs, ne_kwargs, sw_kwargs, se_kwargs, path):
    nw_kwargs = {**nw_kwargs}
    ne_kwargs = {**ne_kwargs}
    sw_kwargs = {**sw_kwargs}
    se_kwargs = {**se_kwargs}

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].imshow(nw_kwargs['data'])
    axs[0, 0].set_title(nw_kwargs['title'])
    axs[0, 0].grid()

    axs[0, 1].imshow(ne_kwargs['data'])
    axs[0, 1].set_title(ne_kwargs['title'])
    axs[0, 1].grid()

    axs[1, 0].imshow(sw_kwargs['data'])
    axs[1, 0].set_title(sw_kwargs['title'])
    axs[1, 0].grid()

    axs[1, 1].imshow(se_kwargs['data'])
    axs[1, 1].set_title(se_kwargs['title'])
    axs[1, 1].grid()

    plt.tight_layout()

    plt.savefig(path)
    plt.close(fig)
