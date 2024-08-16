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
}

# Choose test to simulate
selected_test = 'teste5'

selected_test = panther_tests[selected_test]
recorded_pressure_bscan = np.load(f'{selected_test[0]}/ascan_data.npy')[:, selected_test[4], :, 0].transpose()
time = np.load(f'{selected_test[0]}/time_grid.npy')
sample_time = np.float32((time[1] - time[0]).item())

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

simulation_modes = {
    0: 'TimeReversal',
    1: 'ReverseTimeMigration',
    2: 'TimeReversal + ReverseTimeMigration',
    3: 'Plot last RTM frame',
    4: 'Plot l2-norm TR',
    5: 'TFM',
}

mode = 5

if simulation_modes[mode] == 'TimeReversal':
    tr_sim = TimeReversal(simulation_config, tr_config)
    tr_sim.run(create_animation=True, cmap='bwr')

elif simulation_modes[mode] == 'ReverseTimeMigration':
    rtm_sim = ReverseTimeMigration(simulation_config, rtm_config)
    rtm_sim.run(create_animation=True)

elif simulation_modes[mode] == 'TimeReversal + ReverseTimeMigration':
    tr_sim = TimeReversal(simulation_config, tr_config)
    tr_sim.run(create_animation=True, cmap='bwr', interpolation='nearest', vmax=4e3, vmin=-4e3)

    rtm_sim = ReverseTimeMigration(simulation_config, rtm_config)
    rtm_sim.run(create_animation=True)

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

    plot_imshow(
        data=frame,
        title=f'RTM - Last Frame',
        scatter_kwargs=scatter_kwargs,
        vmax=12000, vmin=-8000,
        interpolation='nearest',
        cmap='bwr',
    )

    save_imshow(
        data=frame,
        title=f'RTM - Last Frame',
        path='./ReverseTimeMigration/last_frame.png',
        scatter_kwargs=scatter_kwargs,
        vmax=12000, vmin=-8000,
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
        vmax=10e4, vmin=7e4,
        interpolation='nearest',
        cmap='bwr',
    )

    save_imshow(
        data=frame,
        title=f'TR - L2 Norm',
        path=f'{tr_sim_folder}/l2-norm.png',
        scatter_kwargs={},
        vmax=10e4, vmin=7e4,
        interpolation='nearest',
        cmap='bwr',
    )

elif simulation_modes[mode] == 'TFM':
    tfm = TFM(selected_test)
    tfm.run()
