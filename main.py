import numpy as np
from AcousticSimulation import AcousticSimulation
from TimeReversal import TimeReversal
from ReverseTimeMigration import ReverseTimeMigration
from plt_utils import plot_imshow, save_imshow
from TFM import TFM

'''
folder,
time (divided by 2) (s),
distance from focus (meters),
speed (m/s),
emitter index,
(grid_size_z, grid_size_x, dz, dx)
'''
panther_tests = {
    'teste1': [
        './panther/teste1_results',
        22.76e-6,
        3.414e-2,
        1500.,
        49,
        (2100, 2100, 3.0e-5, 3.0e-5),
    ],
    'teste2': [
        './panther/teste2_results',
        6.96e-6,
        1.044e-2,
        1500.,
        14,
        (1750, 2100, 2.5e-5, 3.0e-5),
    ],
    'teste3': [
        './panther/teste3_results',
        (7.59e-6, 9.75e-6, 11.97e-6),
        (4.524e-2, 5.811e-2, 7.134e-2),
        5960.,
        (8, 32, 56),
        (1250, 1800, 12.5e-5, 18.0e-5),
    ],
    'teste4': [
        './panther/teste4_results',
        2.96e-6,
        1.9e-2,
        6420.,
        32,
        (1250, 1800, 12.5e-5, 18.0e-5),
    ],
    'teste5': [
        './panther/teste5_results',
        6.08e-6,
        3.903e-2,
        6420.,
        42,
        (1250, 1800, 12.5e-5, 18.0e-5),
    ],
    'teste6': [
        './panther/teste6_results',
        11.44e-6,
        7.344e-2,
        6420.,
        32,
        (1250, 1800, 12.5e-5, 18.0e-5),
    ],
}

simulation_modes = {
    0: 'TimeReversal',
    1: 'ReverseTimeMigration',
    2: 'TimeReversal + ReverseTimeMigration',
    3: 'TFM',
    4: 'Plot last RTM frame',
    5: 'Plot l2-norm TR',
}

# Choose simulation
mode = 3

# Choose test to simulate
selected_test = 'teste6'

# Setup fmc bscan and time array
selected_test = panther_tests[selected_test]

if selected_test[0] == './panther/teste3_results':
    recorded_pressure_bscan = np.load(f'{selected_test[0]}/ascan_data.npy')[:, selected_test[4][1], :, 0].transpose()
else:
    recorded_pressure_bscan = np.load(f'{selected_test[0]}/ascan_data.npy')[:, selected_test[4], :, 0].transpose()
    # recorded_pressure_bscan = np.load(f'{selected_test[0]}/ascan_data.npy')[:, selected_test[4], :, 0]

time = np.load(f'{selected_test[0]}/time_grid.npy')
sample_time = np.float32((time[1] - time[0]).item())

# plot_imshow(np.abs(recorded_pressure_bscan), 'Raw BScan', {}, aspect='auto')

# Panther source needed for RTM
source = np.float32(np.load(f'./panther/panther_source.npy'))

# Add zeros, according to gate_start, to the beginning of array
gate_start_value = np.float32(0)
with open(f'{selected_test[0]}/inspection_params.txt', 'r') as f:
    for line in f:
        if line.startswith('gate_start'):
            gate_start_value = np.float32(line.split('=')[1].strip())
            break
padding_zeros = np.int32(gate_start_value / sample_time)
padding_zeros = np.zeros((len(recorded_pressure_bscan[:, 0]), padding_zeros))

recorded_pressure_bscan = np.hstack((padding_zeros, recorded_pressure_bscan), dtype=np.float32)

# recorded_pressure_bscan[:, 1800:] = np.float32(0)

# plot_imshow(np.abs(recorded_pressure_bscan), 'Raw BScan', {}, aspect='auto')

simulation_config = {
    'dt': sample_time * 1e-6,
    'c': np.full((selected_test[5][0], selected_test[5][1]), selected_test[3]),
    'dz': selected_test[5][2],
    'dx': selected_test[5][3],
    'grid_size_z': selected_test[5][0],
    'grid_size_x': selected_test[5][1],
    'total_time': len(recorded_pressure_bscan[0, :]),
    'animation_step': 100,
}

if selected_test[0] == './panther/teste3_results':
    tr_config = {
        'recorded_pressure_bscan': recorded_pressure_bscan,
        'distance_from_reflector': selected_test[2][1],
        'emitter_index': selected_test[4][1],
        'min_time': 0,
        'max_time': simulation_config['total_time'],
        'padding_zeros': 0,
    }
else:
    tr_config = {
        'recorded_pressure_bscan': recorded_pressure_bscan,
        'distance_from_reflector': selected_test[2],
        'emitter_index': selected_test[4],
        'min_time': 0,
        'max_time': simulation_config['total_time'],
        'padding_zeros': 0,
    }

source = source[:tr_config['max_time']]

rtm_config = {
    'source': source,
}

if simulation_modes[mode] == 'TimeReversal':
    tr_sim = TimeReversal(simulation_config, tr_config)
    tr_sim.run(create_animation=True, cmap='bwr', interpolation='nearest')

elif simulation_modes[mode] == 'ReverseTimeMigration':
    rtm_sim = ReverseTimeMigration(simulation_config, rtm_config)
    rtm_sim.run(create_animation=True)

elif simulation_modes[mode] == 'TimeReversal + ReverseTimeMigration':
    tr_sim = TimeReversal(simulation_config, tr_config)
    tr_sim.run(create_animation=True, cmap='bwr', interpolation='nearest')

    rtm_sim = ReverseTimeMigration(simulation_config, rtm_config)
    rtm_sim.run(create_animation=True)

elif simulation_modes[mode] == 'TFM':
    tfm = TFM(selected_test)
    tfm.run()

elif simulation_modes[mode] == 'Plot last RTM frame':
    tr_sim_folder = './TimeReversal'

    frame = np.load(f'./ReverseTimeMigration/last_frame_rtm/frame_'
                    f'{tr_config['max_time'] - 1}_{np.load(f'{tr_sim_folder}/source_z.npy')}.npy')

    number_of_reflectors = np.load(f'{tr_sim_folder}/number_of_reflectors.npy')
    reflector_z = np.load(f'{tr_sim_folder}/reflector_z.npy')
    reflector_x = np.load(f'{tr_sim_folder}/reflector_x.npy')

    scatter_kwargs = {
        'number_of_reflectors': number_of_reflectors,
        'reflector_z': reflector_z,
        'reflector_x': reflector_x,
    }

    # Receptors setup
    # number_of_receptors = 64
    # receptor_z = []
    # for rp in range(0, number_of_receptors):
    #     receptor_z.append((0.6e-3 * rp) / 12.5e-5)
    # receptor_z = (np.int32(np.asarray(receptor_z)) + np.int32((1250 - receptor_z[-1]) / 2))
    #
    # # Reflectors setup
    # number_of_reflectors = np.int32(3)
    # reflector_z = np.array([receptor_z[8], receptor_z[32], receptor_z[56]], dtype=np.int32)
    # reflector_x = np.array([4.524e-2 / 18.0e-5, 5.811e-2 / 18.0e-5, 7.134e-2 / 18.0e-5], dtype=np.int32)
    #
    # scatter_kwargs = {
    #     'number_of_reflectors': number_of_reflectors,
    #     'reflector_z': reflector_z,
    #     'reflector_x': reflector_x,
    # }
    #
    # f1 = np.load('./frame_4375_511_m8.npy')
    # f2 = np.load('./frame_4375_626_m32.npy')
    # f3 = np.load('./ReverseTimeMigration/last_frame_rtm/frame_4375_741.npy')
    #
    # frame = f1 + f2 + f3

    plot_imshow(
        data=frame,
        title=f'RTM - Last Frame',
        scatter_kwargs=scatter_kwargs,
        vmax=85000, vmin=-85000,
        interpolation='nearest',
        cmap='bwr',
    )

    save_imshow(
        data=frame,
        title=f'RTM - Last Frame',
        path='./ReverseTimeMigration/last_frame.png',
        scatter_kwargs=scatter_kwargs,
        vmax=85000, vmin=-85000,
        interpolation='nearest',
        cmap='bwr',
    )

elif simulation_modes[mode] == 'Plot l2-norm TR':
    tr_sim_folder = './TimeReversal'

    frame = np.load(f'{tr_sim_folder}/l2_norm.npy')

    number_of_reflectors = np.load(f'{tr_sim_folder}/number_of_reflectors.npy')
    reflector_z = np.load(f'{tr_sim_folder}/reflector_z.npy')
    reflector_x = np.load(f'{tr_sim_folder}/reflector_x.npy')

    scatter_kwargs = {
        'number_of_reflectors': number_of_reflectors,
        'reflector_z': reflector_z,
        'reflector_x': reflector_x,
    }

    plot_imshow(
        data=frame,
        title=f'TR - L2 Norm',
        scatter_kwargs={},
        # vmax=10e4, vmin=7e4,
        # interpolation='nearest',
        cmap='bwr',
    )

    save_imshow(
        data=frame,
        title=f'TR - L2 Norm',
        path=f'{tr_sim_folder}/l2-norm.png',
        scatter_kwargs={},
        # vmax=10e4, vmin=7e4,
        # interpolation='nearest',
        cmap='bwr',
    )
