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
    'total_time': 1000,
    'animation_step': 100,
}

ac_config = {
    'source_z': 750,
    'source_x': 2000,
    'mode': 'linear_reflector',  # 'no_reflector', 'punctual_reflector', 'linear_reflector'
    'number_of_reflectors': 1,  # Ignore if 'mode' == 'punctual_reflector' or 'mode' == 'no_reflector'
    'reflector_z': [1200],  # Ignore if 'mode' == 'no_reflector'
    'reflector_x': [2600],  # Ignore if 'mode' == 'no_reflector'
    'reflector_c': 0,
    'number_of_receptors': 3,
    'receptor_z': [250, 500, 750],
    'receptor_x': [2000, 2000, 2000],
}

tr_config = {
    'min_time': 300,
    'max_time': 800,
    'padding_zeros': 500,
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
    ac_sim.run(create_animation=True)
elif simulation_modes[mode] == 'TimeReversal':
    tr_sim = TimeReversal(simulation_config, tr_config)
elif simulation_modes[mode] == 'ReverseTimeMigration':
    rtm_sim = ReverseTimeMigration(**simulation_config)
elif simulation_modes[mode] == 'TimeReversal + ReverseTimeMigration':
    tr_sim = TimeReversal(simulation_config, tr_config)
    rtm_sim = ReverseTimeMigration(**simulation_config)
