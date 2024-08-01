import numpy as np


class SimulationConfig:
    def __init__(self, **simulation_config):
        # Time step (s)
        self.dt = np.float32(simulation_config['dt'])

        # Velocity (m/s)
        self.c = np.float32(simulation_config['c'])

        # Grid Steps (m/px)
        self.dz = np.float32(simulation_config['dz'])
        self.dx = np.float32(simulation_config['dx'])

        # Grid Size (z, x) in pixels
        self.grid_size_z = np.int32(simulation_config['grid_size_z'])
        self.grid_size_x = np.int32(simulation_config['grid_size_x'])

        # Total amount of time steps
        self.total_time = np.int32(simulation_config['total_time'])

        # For creating video
        self.animation_step = np.int32(simulation_config['animation_step'])

        # Simplify typing
        self.grid_size_shape = (self.grid_size_z, self.grid_size_x)

        print(f'Total time: {self.total_time}')

        print(f'CFL-Z: {self.c * (self.dt / self.dz)}')
        print(f'CFL-X: {self.c * (self.dt / self.dx)}')

        print(f'Grid Size (px): ({self.grid_size_z}, {self.grid_size_x})')

        self.p_future = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.p_present = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.p_past = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.lap = np.zeros(self.grid_size_shape, dtype=np.float32)
