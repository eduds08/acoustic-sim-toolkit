import numpy as np
import os
from WebGPUConfig import WebGPUConfig


class ReverseTimeMigration(WebGPUConfig):
    def __init__(self, **simulation_config):
        super().__init__(**simulation_config)

        self.shader_file = './reverse_time_migration.wgsl'

        self.folder = './ReverseTimeMigration'
        self.last_frame_rtm_folder = f'{self.folder}/last_frame_rtm'
        self.animation_folder = f'{self.folder}/animation'

        os.makedirs(self.folder, exist_ok=True)
        os.makedirs(self.last_frame_rtm_folder, exist_ok=True)
        os.makedirs(self.animation_folder, exist_ok=True)

        self.tr_sim_folder = './TimeReversal'
        self.tr_last_frames_folder = f'{self.tr_sim_folder}/last_frames'

        self.ac_sim_folder = './AcousticSimulation'
        self.ac_source_folder = f'{self.ac_sim_folder}/source_setup'

        self.rtm_total_time = np.load(f'{self.tr_sim_folder}/tr_total_time.npy')

        print(f'Total time (RTM): {self.rtm_total_time}')

        # Source setup
        self.source_z = np.load(f'{self.ac_source_folder}/source_z.npy')
        self.source_x = np.load(f'{self.ac_source_folder}/source_x.npy')
        self.source = np.load(f'{self.ac_source_folder}/source.npy')

        self.p_future_reversed_tr = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.p_present_reversed_tr = np.load(f'{self.tr_last_frames_folder}/tr_{self.rtm_total_time - 2}.npy')
        self.p_past_reversed_tr = np.load(f'{self.tr_last_frames_folder}/tr_{self.rtm_total_time - 1}.npy')
        self.lap_reversed_tr = np.zeros(self.grid_size_shape, dtype=np.float32)
