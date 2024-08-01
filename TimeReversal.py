import numpy as np
import os
from WebGPUConfig import WebGPUConfig


class TimeReversal(WebGPUConfig):
    def __init__(self, simulation_config, tr_config):
        super().__init__(**simulation_config)

        tr_config = {**tr_config}

        self.shader_file = './time_reversal.wgsl'

        self.folder = './TimeReversal'
        self.last_frames = f'{self.folder}/last_frames'
        self.animation_folder = f'{self.folder}/animation'

        os.makedirs(self.folder, exist_ok=True)
        os.makedirs(self.last_frames, exist_ok=True)
        os.makedirs(self.animation_folder, exist_ok=True)

        self.ac_sim_folder = './AcousticSimulation'
        self.ac_receptors_folder = f'{self.ac_sim_folder}/receptors_setup'
        self.ac_recorded_pressure_folder = f'{self.ac_sim_folder}/recorded_pressure'

        # Receptors setup
        self.number_of_receptors = np.load(f'{self.ac_receptors_folder}/number_of_receptors.npy')
        self.receptor_z = np.load(f'{self.ac_receptors_folder}/receptor_z.npy')
        self.receptor_x = np.load(f'{self.ac_receptors_folder}/receptor_x.npy')

        self.min_time = np.int32(tr_config['min_time'])
        self.max_time = np.int32(tr_config['max_time'])
        self.padding_zeros = np.int32(tr_config['padding_zeros'])

        self.tr_total_time = np.int32(self.padding_zeros + (self.max_time - self.min_time))
        np.save(f'{self.folder}/tr_total_time.npy', self.tr_total_time)

        print(f'Total time (TR): {self.tr_total_time}')

        self.reversed_pressure = []

        # Recorded pressure on receptors
        for i in range(self.number_of_receptors):
            recorded_pressure = np.load(
                f'{self.ac_recorded_pressure_folder}/receptor_{i}.npy'
            )[self.min_time:self.max_time]

            flipped_recorded_pressure = np.array(np.flip(recorded_pressure), dtype=np.float32)

            padded_recorded_pressure = np.pad(flipped_recorded_pressure, (0, self.padding_zeros), mode='constant')

            self.reversed_pressure.append(padded_recorded_pressure)
