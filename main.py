import numpy as np
from TimeReversal import TimeReversal
from plt_utils import plot_imshow
from scipy.io import loadmat
import matplotlib.pyplot as plt

speed = 1500.
# grid_size_z, grid_size_x, dz, dx
grid = (3000, 3000, 3.e-1, 3.e-1)  # 300x300 (metros)

acude = loadmat('./acude/azulPerpendicular1_Variables.mat')

recorded_pressure_bscan = acude['data'].transpose()
dstep = acude['dstep']

repRate = acude['repRate']
sample_time = np.float32((1 / repRate).item())

# Add zeros, according to gate_start, to the beginning of array
gate_start_value = np.float32(0)
padding_zeros = np.int32(gate_start_value / sample_time)
padding_zeros = np.zeros((len(recorded_pressure_bscan[:, 0]), padding_zeros))

recorded_pressure_bscan = np.hstack((padding_zeros, recorded_pressure_bscan), dtype=np.float32)

recorded_pressure_bscan = recorded_pressure_bscan[:, :50000]
plot_imshow(np.abs(recorded_pressure_bscan), 'Raw BScan', {}, aspect='auto')

simulation_config = {
    'dt': sample_time,
    'c': np.full((grid[0], grid[1]), speed),
    'dz': grid[2],
    'dx': grid[3],
    'grid_size_z': grid[0],
    'grid_size_x': grid[1],
    'total_time': len(recorded_pressure_bscan[0, :]),
    'animation_step': 500,
}

tr_config = {
    'recorded_pressure_bscan': recorded_pressure_bscan,
    'min_time': 0,
    'max_time': simulation_config['total_time'],
    'padding_zeros': 0,
    'recs_dist': dstep,
}

norma = np.load('./TimeReversal/l2_norm.npy')

plt.figure()
plt.imshow(norma, aspect='auto')
plt.colorbar()
plt.grid()
plt.title('L2-Norm')
plt.show()

# tr_sim = TimeReversal(simulation_config, tr_config)
# tr_sim.run(create_animation=False, cmap='bwr')
