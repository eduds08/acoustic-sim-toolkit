import numpy as np
import os
from WebGPUConfig import WebGPUConfig
from plt_utils import save_imshow, save_imshow_4_subplots
from os_utils import clear_folder, create_ffmpeg_animation


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

        self.info_int = np.array(
            [
                self.grid_size_z,
                self.grid_size_x,
                self.source_z,
                self.source_x,
                self.number_of_reflectors,
                0,
            ],
            dtype=np.int32
        )

        self.info_float = np.array(
            [
                self.dz,
                self.dx,
                self.dt,
                self.c,
                self.reflector_c,
            ],
            dtype=np.float32
        )

    def run(self, create_animation: bool):
        shader_file = open(self.shader_file)
        shader_string = shader_file.read().replace('wsz', f'{self.wsz}').replace('wsx', f'{self.wsx}')
        shader_file.close()

        self.shader_module = self.device.create_shader_module(code=shader_string)

        wgsl_data = {
            'infoI32': self.info_int,
            'infoF32': self.info_float,
            'receptor_z': self.receptor_z,
            'receptor_x': self.receptor_x,
            'p_future': self.p_future,
            'p_present': self.p_present,
            'p_past': self.p_past,
            'lap': self.lap,
        }

        for i in range(self.number_of_receptors):
            wgsl_data[f'reversed_pressure_{i}'] = self.reversed_pressure[i]

        shader_lines = list(shader_string.split('\n'))
        buffers = self.create_buffers(wgsl_data, shader_lines)

        compute_lap = self.create_compute_pipeline("laplacian_5_operator")
        compute_sim = self.create_compute_pipeline("sim")
        compute_incr = self.create_compute_pipeline("incr_time")

        if create_animation:
            clear_folder(self.animation_folder)

        for i in range(self.tr_total_time):
            command_encoder = self.device.create_command_encoder()
            compute_pass = command_encoder.begin_compute_pass()

            for index, bind_group in enumerate(self.bind_groups):
                compute_pass.set_bind_group(index, bind_group, [], 0, 999999)

            compute_pass.set_pipeline(compute_lap)
            compute_pass.dispatch_workgroups(self.grid_size_z // self.wsz,
                                             self.grid_size_x // self.wsx)

            compute_pass.set_pipeline(compute_sim)
            compute_pass.dispatch_workgroups(self.grid_size_z // self.wsz,
                                             self.grid_size_x // self.wsx)

            compute_pass.set_pipeline(compute_incr)
            compute_pass.dispatch_workgroups(1)

            compute_pass.end()
            self.device.queue.submit([command_encoder.finish()])

            """ READ BUFFERS """
            self.p_future = (np.asarray(self.device.queue.read_buffer(buffers['b4']).cast("f"))
                             .reshape(self.grid_size_shape))

            if i == self.tr_total_time - 1 or i == self.tr_total_time - 2:
                np.save(f'{self.last_frames_folder}/tr_{i}', self.p_future)

            if i % self.animation_step == 0:
                if create_animation:
                    save_imshow(
                        data=self.p_future,
                        title=f'Time Reversal',
                        path=f'{self.animation_folder}/plot_{i}.png',
                        scatter_kwargs={
                            'number_of_receptors': self.number_of_receptors,
                            'receptor_z': self.receptor_z,
                            'receptor_x': self.receptor_x,
                        },
                        plt_kwargs={
                            # 'vmax': 1e-3,
                            # 'vmin': -1e-3,
                        },
                        plt_grid=True,
                        plt_colorbar=True,
                    )
                print(f'Time Reversal - i={i}')

        print('Time Reversal finished.')

        if create_animation:
            create_ffmpeg_animation(self.animation_folder, 'tr.mkv', self.tr_total_time, self.animation_step)
