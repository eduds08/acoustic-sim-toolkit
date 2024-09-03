import numpy as np
from TimeReversal import TimeReversal
import mat73
import matplotlib.pyplot as plt
from plt_utils import plot_imshow

b = mat73.loadmat('DAS_POOL/FMC_3_1_50ns/FMC_3_1_50ns_variables.mat')

time = np.float32(b['time'])
sample_time = np.float32((time[1] - time[0]).item())

recorded_pressure_bscan = np.zeros((10, 100000))
for c in range(10):
    recorded_pressure_bscan[c] = b[f'signalMic{c + 1}']

gate_start_value = np.float32(0)
padding_zeros = np.int32(gate_start_value / sample_time)
padding_zeros = np.zeros((len(recorded_pressure_bscan[:, 0]), padding_zeros))

recorded_pressure_bscan = np.hstack((padding_zeros, recorded_pressure_bscan), dtype=np.float32)

recorded_pressure_bscan = np.float32(recorded_pressure_bscan[:, 22000:30000])

plot_imshow(np.abs(recorded_pressure_bscan), 'Raw BScan', {}, aspect='auto')

grid_size_z = 1500
grid_size_x = 2500

simulation_config = {
    'dt': sample_time,
    'c': np.full((grid_size_z, grid_size_x), np.float32(1500.)),
    'dz': np.float32(1),
    'dx': np.float32(1),
    'grid_size_z': grid_size_z,
    'grid_size_x': grid_size_x,
    'total_time': len(recorded_pressure_bscan[0, :]),
    'animation_step': 100,
}

tr_config = {
    'recorded_pressure_bscan': recorded_pressure_bscan,
    'min_time': 0,
    'max_time': simulation_config['total_time'],
    'padding_zeros': 0,
}

tr_sim = TimeReversal(simulation_config, tr_config)
tr_sim.run(create_animation=False)

l2_norm = np.load('./TimeReversal/l2_norm.npy')
plt.figure()
plt.imshow(l2_norm)
plt.colorbar()
plt.grid()
plt.show()
