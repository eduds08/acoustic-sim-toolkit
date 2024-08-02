from AcousticSimulation import AcousticSimulation
from TimeReversal import TimeReversal
from ReverseTimeMigration import ReverseTimeMigration

simulation_config = {
    'dt': 1e-3,
    'c': 1500,
    'dz': 3,
    'dx': 3,
    'grid_size_z': 4000,
    'grid_size_x': 4000,
    'total_time': 4300,
    'animation_step': 100,
}

ac_config = {
    'source_z': 1000,
    'source_x': 2000,
    'mode': 'linear_reflector',  # 'no_reflector', 'punctual_reflector', 'linear_reflector'
    'number_of_reflectors': 500,  # Ignore if 'mode' == 'punctual_reflector' or 'mode' == 'no_reflector'
    'reflector_z': [2200 + i for i in range(500)],  # Ignore if 'mode' == 'no_reflector'
    'reflector_x': [2600 for _ in range(500)],  # Ignore if 'mode' == 'no_reflector'
    'reflector_c': 0,
    'number_of_receptors': 7,
    'receptor_z': [250 + 1000, 500 + 1000, 750 + 1000, 1000 + 1000, 1250 + 1000, 1500 + 1000, 1750 + 1000],
    'receptor_x': [2000 for _ in range(7)],
}

tr_config = {
    'min_time': 0,
    'max_time': 4300,
    'padding_zeros': 0,
}

simulation_modes = {
    0: 'AcousticSimulation',
    1: 'TimeReversal',
    2: 'ReverseTimeMigration',
    3: 'TimeReversal + ReverseTimeMigration',
}

mode = 1

if simulation_modes[mode] == 'AcousticSimulation':
    ac_sim = AcousticSimulation(simulation_config, ac_config)
    ac_sim.run(create_animation=True)
elif simulation_modes[mode] == 'TimeReversal':
    tr_sim = TimeReversal(simulation_config, tr_config)
    tr_sim.run(create_animation=True)
elif simulation_modes[mode] == 'ReverseTimeMigration':
    rtm_sim = ReverseTimeMigration(**simulation_config)
elif simulation_modes[mode] == 'TimeReversal + ReverseTimeMigration':
    tr_sim = TimeReversal(simulation_config, tr_config)
    rtm_sim = ReverseTimeMigration(**simulation_config)
