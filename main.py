from AcousticSimulation import AcousticSimulation
from TimeReversal import TimeReversal
from ReverseTimeMigration import ReverseTimeMigration

simulation_config = {
    'dt': 1e-3,
    'c': 1500,
    'dz': 3,
    'dx': 3,
    'grid_size_z': 1000,
    'grid_size_x': 1000,
    'total_time': 1500,
    'animation_step': 10,
}

ac_config = {
    'source_z': 400,
    'source_x': 500,
    'mode': 'linear_reflector',  # 'no_reflector', 'punctual_reflector', 'linear_reflector'
    'number_of_reflectors': 100,  # Ignore if 'mode' == 'punctual_reflector' or 'mode' == 'no_reflector'
    'reflector_z': [500 - i for i in range(50)] + [501 + i for i in range(50)],  # Ignore if 'mode' == 'no_reflector'
    'reflector_x': [650 - i for i in range(50)] + [649 - i for i in range(50)],  # Ignore if 'mode' == 'no_reflector'
    'reflector_c': 0,
    'number_of_receptors': 5,
    'receptor_z': [400 + (50 * i) for i in range(5)],
    'receptor_x': [500 for _ in range(5)],
}

tr_config = {
    'min_time': 550,
    'max_time': 1250,
    'padding_zeros': 550,
}

simulation_modes = {
    0: 'AcousticSimulation',
    1: 'TimeReversal',
    2: 'ReverseTimeMigration',
    3: 'TimeReversal + ReverseTimeMigration',
}

mode = 0

if simulation_modes[mode] == 'AcousticSimulation':
    ac_sim = AcousticSimulation(simulation_config, ac_config)
    ac_sim.run(create_animation=True, plt_kwargs={})
elif simulation_modes[mode] == 'TimeReversal':
    tr_sim = TimeReversal(simulation_config, tr_config)
    tr_sim.run(create_animation=True, plt_kwargs={})
elif simulation_modes[mode] == 'ReverseTimeMigration':
    rtm_sim = ReverseTimeMigration(**simulation_config)
    rtm_sim.run(create_animation=True, plt_kwargs={})
elif simulation_modes[mode] == 'TimeReversal + ReverseTimeMigration':
    tr_sim = TimeReversal(simulation_config, tr_config)
    tr_sim.run(create_animation=True, plt_kwargs={})

    rtm_sim = ReverseTimeMigration(**simulation_config)
    rtm_sim.run(create_animation=True, plt_kwargs={})
