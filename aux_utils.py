import numpy as np
from matplotlib.image import imread


def convert_image_to_matrix(image_path):
    rgb_raw_image = np.int32(imread(image_path))

    velocity_map = {
        'white': 'receptors',
        'black': '0',
        'blue': '1500',
        'green': '3200',
        'red': '6400',
    }
    binary_color = {
        7: 'white',
        0: 'black',
        1: 'red',
        2: 'green',
        4: 'blue',
    }

    rgb_2d_grid = np.zeros_like(rgb_raw_image[:, :, 0])

    b = 1
    for i in range(3):
        b += i
        rgb_2d_grid += rgb_raw_image[:, :, i] * b

    rgb_string = np.array(rgb_2d_grid, dtype='str')
    for k in binary_color.keys():
        rgb_string[rgb_string == str(k)] = velocity_map[binary_color[k]]

    receptor_pos = np.where(rgb_string == 'receptors')
    rgb_string[receptor_pos] = '1500'
    receptor_z, receptor_x = receptor_pos

    rgb_float = np.array(rgb_string, dtype=np.float32)

    return rgb_float, receptor_z, receptor_x
