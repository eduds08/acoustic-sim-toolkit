import numpy as np
from AcousticSimulation import AcousticSimulation
from TimeReversal import TimeReversal
from ReverseTimeMigration import ReverseTimeMigration
from plt_utils import plot_imshow, save_imshow
from TFM import TFM
import matplotlib.pyplot as plt

'''
folder,
distance from focus (meters),
speed (m/s),
(grid_size_z, grid_size_x, dz, dx)
'''
panther_tests = {
    'teste1': [
        './panther/teste1_results',
        3.414e-2,
        1500.,
        (2100, 2100, 3.0e-5, 3.0e-5),
    ],
    'teste2': [
        './panther/teste2_results',
        1.044e-2,
        1500.,
        (1750, 2100, 2.5e-5, 3.0e-5),
    ],
    'teste3': [
        './panther/teste3_results',
        1.04e-2,
        1500.,
        (2000, 2000, 6e-5, 6e-5),
    ],
    'teste4': [
        './panther/teste4_results',
        1.9e-2,
        6420.,
        (1250, 1800, 12.5e-5, 18.0e-5),
    ],
    'teste5': [
        './panther/teste5_results',
        3.903e-2,
        6420.,
        (1250, 1800, 12.5e-5, 18.0e-5),
    ],
    'teste6': [
        './panther/teste6_results',
        7.344e-2,
        6420.,
        (1250, 1800, 12.5e-5, 18.0e-5),
    ],
    'teste7': [
        './panther/teste7_results',
        0,
        6420.,
        (1250, 1800, 12.5e-5, 18.0e-5),
    ],
}

simulation_modes = {
    0: 'TimeReversal',
    1: 'ReverseTimeMigration',
    2: 'TimeReversal + ReverseTimeMigration',
    3: 'TFM',
}

# Choose simulation
mode = 2

# Choose test to simulate
selected_test = 'teste1'

# Setup fmc bscan and time array
selected_test = panther_tests[selected_test]

for c in range(64):
    recorded_pressure_bscan = np.load(f'{selected_test[0]}/ascan_data.npy')[:, c, :, 0].transpose()

    time = np.load(f'{selected_test[0]}/time_grid.npy')
    sample_time = np.float32((time[1] - time[0]).item())

    # plot_imshow(np.abs(recorded_pressure_bscan), 'Raw BScan', {}, aspect='auto')

    # Panther source needed for RTM
    source = np.float32(np.load(f'./panther/source_for_rtm.npy'))

    # Add zeros, according to gate_start, to the beginning of array
    gate_start_value = np.float32(0)
    if selected_test[0] != './panther/teste7_results':
        with open(f'{selected_test[0]}/inspection_params.txt', 'r') as f:
            for line in f:
                if line.startswith('gate_start'):
                    gate_start_value = np.float32(line.split('=')[1].strip())
                    break
    padding_zeros = np.int32(gate_start_value / sample_time)
    padding_zeros = np.zeros((len(recorded_pressure_bscan[:, 0]), padding_zeros))

    recorded_pressure_bscan = np.hstack((padding_zeros, recorded_pressure_bscan), dtype=np.float32)

    # recorded_pressure_bscan[:, 6400:] = np.float32(0)
    plot_imshow(np.abs(recorded_pressure_bscan), 'Raw BScan', {}, aspect='auto')

    simulation_config = {
        'dt': sample_time * 1e-6,
        'c': np.full((selected_test[3][0], selected_test[3][1]), selected_test[2]),
        'dz': selected_test[3][2],
        'dx': selected_test[3][3],
        'grid_size_z': selected_test[3][0],
        'grid_size_x': selected_test[3][1],
        'total_time': len(recorded_pressure_bscan[0, :]),
        'animation_step': 100,
    }

    tr_config = {
        'recorded_pressure_bscan': recorded_pressure_bscan,
        'distance_from_reflector': selected_test[1],
        'emitter_index': c,
        'min_time': 0,
        'max_time': simulation_config['total_time'],
        'padding_zeros': 0,
    }

    source = np.pad(source, ((0, tr_config['max_time'] - len(source)), (0, 0)), mode='constant', constant_values=0)

    rtm_config = {
        'source': source,
        'emitter_index': c,
        'test': selected_test[0],
    }

    if simulation_modes[mode] == 'TimeReversal':
        tr_sim = TimeReversal(simulation_config, tr_config)
        tr_sim.run(create_animation=True, cmap='bwr')

    elif simulation_modes[mode] == 'ReverseTimeMigration':
        rtm_sim = ReverseTimeMigration(simulation_config, rtm_config)
        rtm_sim.run(create_animation=True)

    elif simulation_modes[mode] == 'TimeReversal + ReverseTimeMigration':
        tr_sim = TimeReversal(simulation_config, tr_config)
        tr_sim.run(create_animation=False, cmap='bwr')

        rtm_sim = ReverseTimeMigration(simulation_config, rtm_config)
        rtm_sim.run(create_animation=False)

    elif simulation_modes[mode] == 'TFM':
        tfm = TFM(selected_test)
        tfm.run()
