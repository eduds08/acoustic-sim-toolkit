import numpy as np

from AcousticSimulation import AcousticSimulation
from TimeReversal import TimeReversal
from ReverseTimeMigration import ReverseTimeMigration
from aux_utils import convert_image_to_matrix
import matplotlib.pyplot as plt

velocity_map, receptor_z, receptor_x = convert_image_to_matrix('map.png')

simulation_config = {
    'dt': 1e-3,
    'c': velocity_map,
    'dz': 3,
    'dx': 3,
    'grid_size_z': len(velocity_map[:, 0]),
    'grid_size_x': len(velocity_map[0, :]),
    'total_time': 3500,
    'animation_step': 100,
}

ac_config = {
    'source_z': receptor_z[0],
    'source_x': receptor_x[0],
    'number_of_receptors': len(receptor_z),
    'receptor_z': receptor_z,
    'receptor_x': receptor_x,
}

tr_config = {
    'min_time': 0,
    'max_time': simulation_config['total_time'],
    'padding_zeros': 0,
}

simulation_modes = {
    0: 'AcousticSimulation',
    1: 'TimeReversal',
    2: 'ReverseTimeMigration',
    3: 'TimeReversal + ReverseTimeMigration',
    4: 'Full',
}

mode = 4

if simulation_modes[mode] == 'AcousticSimulation':
    ac_sim = AcousticSimulation(simulation_config, ac_config)
    ac_sim.run(create_animation=True, plt_kwargs={'cmap': 'bwr'})
elif simulation_modes[mode] == 'TimeReversal':
    tr_sim = TimeReversal(simulation_config, tr_config)
    tr_sim.run(create_animation=True, plt_kwargs={'cmap': 'bwr'})
elif simulation_modes[mode] == 'ReverseTimeMigration':
    rtm_sim = ReverseTimeMigration(**simulation_config)
    rtm_sim.run(create_animation=True, plt_kwargs={'cmap': 'bwr'})
elif simulation_modes[mode] == 'TimeReversal + ReverseTimeMigration':
    tr_sim = TimeReversal(simulation_config, tr_config)
    tr_sim.run(create_animation=True, plt_kwargs={})

    rtm_sim = ReverseTimeMigration(**simulation_config)
    rtm_sim.run(create_animation=True, plt_kwargs={})
elif simulation_modes[mode] == 'Full':
    ac_sim = AcousticSimulation(simulation_config, ac_config)
    ac_sim.run(create_animation=True, plt_kwargs={'cmap': 'bwr'})
    tr_sim = TimeReversal(simulation_config, tr_config)
    tr_sim.run(create_animation=True, plt_kwargs={'cmap': 'bwr'})
    rtm_sim = ReverseTimeMigration(**simulation_config)
    rtm_sim.run(create_animation=True, plt_kwargs={'cmap': 'bwr'})
