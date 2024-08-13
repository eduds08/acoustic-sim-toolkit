import numpy as np
from AcousticSimulation import AcousticSimulation
from TimeReversal import TimeReversal
from ReverseTimeMigration import ReverseTimeMigration
from plt_utils import plot_imshow, plot_imshow_2
import matplotlib.pyplot as plt

simulation_config = {
    'dt': 8e-9,
    'c': 1500,
    'dz': 2.5e-5,
    'dx': 3.0e-5,
    'grid_size_z': 1750,
    'grid_size_x': 2100,
    'total_time': 3750,
    'animation_step': 100,
}

simulation_modes = {
    1: 'TimeReversal',
    2: 'ReverseTimeMigration',
    3: 'TimeReversal + ReverseTimeMigration',
}

mode = 1

if simulation_modes[mode] == 'TimeReversal':
    tr_sim = TimeReversal(current_rec=49, **simulation_config)
    tr_sim.run(create_animation=True, plt_kwargs={})
elif simulation_modes[mode] == 'ReverseTimeMigration':
    rtm_sim = ReverseTimeMigration(current_rec=24, **simulation_config)
    rtm_sim.run(create_animation=True, plt_kwargs={})

elif simulation_modes[mode] == 'TimeReversal + ReverseTimeMigration':
    tr_sim = TimeReversal(current_rec=49, **simulation_config)
    tr_sim.run(create_animation=True, plt_kwargs={
        'vmax': 4e2,
        'vmin': -4e2,
    })

    rtm_sim = ReverseTimeMigration(current_rec=24, **simulation_config)
    rtm_sim.run(create_animation=True, plt_kwargs={})
