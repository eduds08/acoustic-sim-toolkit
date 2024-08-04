import numpy as np
import os
from WebGPUConfig import WebGPUConfig
from plt_utils import save_imshow, save_imshow_4_subplots, normal_plot
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

        clear_folder(self.last_frame_rtm_folder)

        self.tr_sim_folder = './TimeReversal'
        self.tr_last_frames_folder = f'{self.tr_sim_folder}/last_frames'

        self.rtm_total_time = np.load(f'{self.tr_sim_folder}/tr_total_time.npy')

        print(f'Total time (RTM): {self.rtm_total_time}')

        # Receptors setup
        receptor_z = []
        for rp in range(0, 64):
            receptor_z.append((6.0e-4 * rp) / simulation_config['dz'])

        self.number_of_receptors = np.int32(64)
        self.receptor_z = (np.int32(np.asarray(receptor_z)) + np.int32((simulation_config['grid_size_z']
                                                                        - receptor_z[-1]) / 2))
        self.receptor_x = np.array([2000 for _ in range(64)])

        # Reflectors setup
        self.number_of_reflectors = np.int32(5)
        self.reflector_z = self.receptor_z[np.asarray([0, 10, 32, 54, 63])]
        self.reflector_x = np.int32(np.asarray([4.139e-2, 4.592e-2, 5.796e-2, 6.995e-2, 7.500e-2]) / self.dx)

        # Source setup
        self.source_z = np.int32(self.receptor_z[0])
        self.source_x = np.int32(self.receptor_x[0])
        self.source = np.float32(np.load('./panther/source.npy'))

        # zeros = np.zeros((450, 1), dtype=np.float32)
        # self.source = np.concatenate((zeros, self.source), axis=0)
        self.source = self.source[:self.rtm_total_time]

        normal_plot(self.source)

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
            ],
            dtype=np.float32
        )

    def run(self, create_animation: bool, plt_kwargs=None):
        if plt_kwargs is None:
            plt_kwargs = {}

        shader_file = open(self.shader_file)
        shader_string = shader_file.read().replace('wsz', f'{self.wsz}').replace('wsx', f'{self.wsx}')
        shader_file.close()

        self.shader_module = self.device.create_shader_module(code=shader_string)

        wgsl_data = {
            'infoI32': self.info_int,
            'infoF32': self.info_float,
            'source': self.source,
            'p_future': self.p_future,
            'p_present': self.p_present,
            'p_past': self.p_past,
            'lap': self.lap,
            'p_future_reversed_tr': self.p_future_reversed_tr,
            'p_present_reversed_tr': self.p_present_reversed_tr,
            'p_past_reversed_tr': self.p_past_reversed_tr,
            'lap_reversed_tr': self.lap_reversed_tr,
        }

        shader_lines = list(shader_string.split('\n'))
        buffers = self.create_buffers(wgsl_data, shader_lines)

        compute_lap = self.create_compute_pipeline("laplacian_5_operator")
        compute_sim_reversed_tr = self.create_compute_pipeline("sim_reversed_tr")
        compute_sim = self.create_compute_pipeline("sim")
        compute_incr = self.create_compute_pipeline("incr_time")

        if create_animation:
            clear_folder(self.animation_folder)

        scatter_kwargs = {
            'number_of_reflectors': self.number_of_reflectors,
            'reflector_z': self.reflector_z,
            'reflector_x': self.reflector_x + np.int32(2000),
        }

        accumulated_product = np.zeros_like(self.p_future)

        for i in range(self.rtm_total_time):
            command_encoder = self.device.create_command_encoder()
            compute_pass = command_encoder.begin_compute_pass()

            for index, bind_group in enumerate(self.bind_groups):
                compute_pass.set_bind_group(index, bind_group, [], 0, 999999)

            compute_pass.set_pipeline(compute_lap)
            compute_pass.dispatch_workgroups(self.grid_size_z // self.wsz,
                                             self.grid_size_x // self.wsx)

            compute_pass.set_pipeline(compute_sim_reversed_tr)
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
            self.p_future = (np.asarray(self.device.queue.read_buffer(buffers['b3']).cast("f"))
                             .reshape(self.grid_size_shape))

            self.p_future_reversed_tr = (np.asarray(self.device.queue.read_buffer(buffers['b7']).cast("f"))
                                         .reshape(self.grid_size_shape))

            current_product = self.p_future * self.p_future_reversed_tr
            accumulated_product += current_product

            if i == self.rtm_total_time - 1 or i == 2100:
                save_imshow(
                    data=accumulated_product,
                    title=f'RTM - Last Frame',
                    path=f'{self.last_frame_rtm_folder}/plot_{i}.png',
                    scatter_kwargs=scatter_kwargs,
                    plt_kwargs=plt_kwargs,
                    plt_grid=False,
                    plt_colorbar=True,
                )
                np.save(f'{self.last_frame_rtm_folder}/frame_{i}.npy', accumulated_product)

            if i % self.animation_step == 0:
                if create_animation:
                    save_imshow_4_subplots(
                        nw_kwargs={'data': self.p_future_reversed_tr, 'title': 'Up-going wavefields', 'plt_grid': True},
                        ne_kwargs={'data': current_product, 'title': 'Product (Down * Up)', 'plt_grid': False},
                        sw_kwargs={'data': self.p_future, 'title': 'Down-going wavefields', 'plt_grid': True},
                        se_kwargs={'data': accumulated_product, 'title': 'Accumulated product', 'plt_grid': False},
                        path=f'{self.animation_folder}/plot_{i}.png',
                    )

            if i % 100 == 0:
                print(f'Reverse Time Migration - i={i}')

        print('Reverse Time Migration finished.')

        if create_animation:
            create_ffmpeg_animation(self.animation_folder, 'rtm.mp4', self.rtm_total_time, self.animation_step)
