import numpy as np
from AcousticSimulation import AcousticSimulation
from TimeReversal import TimeReversal
from ReverseTimeMigration import ReverseTimeMigration

simulation_config = {
    'dt': 8e-9,
    'c': 5960,
    'dz': 12.5e-5,
    'dx': 18.e-5,
    'grid_size_z': 3500,
    'grid_size_x': 5040,
    'total_time': 3750,
    'animation_step': 100,
}

receptor_z = []
for rp in range(0, 64):
    receptor_z.append((6.0e-4 * rp) / simulation_config['dz'])

simulation_modes = {
    1: 'TimeReversal',
    2: 'ReverseTimeMigration',
    3: 'TimeReversal + ReverseTimeMigration',
}

mode = 3

if simulation_modes[mode] == 'TimeReversal':
    tr_sim = TimeReversal(simulation_config)
    tr_sim.run(create_animation=True, plt_kwargs={
        'vmax': 1e3,
        'vmin': -1e3,
    })
elif simulation_modes[mode] == 'ReverseTimeMigration':
    rtm_sim = ReverseTimeMigration(**simulation_config)
    rtm_sim.run(create_animation=True, plt_kwargs={
        'vmax': 1e5,
        'vmin': -1e5,
    })
elif simulation_modes[mode] == 'TimeReversal + ReverseTimeMigration':
    tr_sim = TimeReversal(simulation_config)
    tr_sim.run(create_animation=True, plt_kwargs={
        'vmax': 1e3,
        'vmin': -1e3,
    })

    rtm_sim = ReverseTimeMigration(**simulation_config)
    rtm_sim.run(create_animation=True, plt_kwargs={
        'vmax': 1e5,
        'vmin': -1e5,
    })
