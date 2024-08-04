import numpy as np
from AcousticSimulation import AcousticSimulation
from TimeReversal import TimeReversal
from ReverseTimeMigration import ReverseTimeMigration
from plt_utils import plot_imshow, plot_imshow_2
import matplotlib.pyplot as plt

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

simulation_modes = {
    1: 'TimeReversal',
    2: 'ReverseTimeMigration',
    3: 'TimeReversal + ReverseTimeMigration',
}

mode = 3

# Receptors setup
receptor_z = []
for rp in range(0, 64):
    receptor_z.append((6.0e-4 * rp) / simulation_config['dz'])

receptor_z = np.int32(np.int32(np.asarray(receptor_z)) + np.int32((simulation_config['grid_size_z']
                                                                   - receptor_z[-1]) / 2))

# Reflectors setup
number_of_reflectors = np.int32(5)
reflector_z = receptor_z[np.asarray([0, 10, 32, 54, 63])]
reflector_x = np.int32(np.asarray([4.139e-2, 4.592e-2, 5.796e-2, 6.995e-2, 7.500e-2]) / np.float32(18.e-5)) + np.int32(2000)

print(reflector_z)
print(reflector_x)

frame = np.load('./frame_2100.npy')

sub_matriz = frame[1400:1700, 2100:2350]

sub_matriz[198, 129] = 1e10

plt.figure()
plt.imshow(sub_matriz, cmap='bwr', vmax=4.5e4, vmin=-4e4)
plt.colorbar()
plt.grid()
plt.show()

# if simulation_modes[mode] == 'TimeReversal':
#     tr_sim = TimeReversal(simulation_config)
#     tr_sim.run(create_animation=True, plt_kwargs={
#         'vmax': 1e3,
#         'vmin': -1e3,
#     })
# elif simulation_modes[mode] == 'ReverseTimeMigration':
#     rtm_sim = ReverseTimeMigration(**simulation_config)
#     rtm_sim.run(create_animation=True, plt_kwargs={
#         'vmax': 1e5,
#         'vmin': -1e5,
#     })
# elif simulation_modes[mode] == 'TimeReversal + ReverseTimeMigration':
#     tr_sim = TimeReversal(simulation_config)
#     tr_sim.run(create_animation=True, plt_kwargs={
#         'vmax': 4e2,
#         'vmin': -4e2,
#     })
#
#     rtm_sim = ReverseTimeMigration(**simulation_config)
#     rtm_sim.run(create_animation=True, plt_kwargs={
#         'vmax': 4e2,
#         'vmin': -4e2,
#     })
