import numpy as np
import os
from WebGPUConfig import WebGPUConfig
from plt_utils import save_imshow, plot_imshow
from os_utils import clear_folder, create_ffmpeg_animation


class TimeReversal(WebGPUConfig):
    def __init__(self, simulation_config):
        super().__init__(**simulation_config)

        self.shader_file = './time_reversal.wgsl'

        self.folder = './TimeReversal'
        self.last_frames_folder = f'{self.folder}/last_frames'
        self.animation_folder = f'{self.folder}/animation'

        os.makedirs(self.folder, exist_ok=True)
        os.makedirs(self.last_frames_folder, exist_ok=True)
        os.makedirs(self.animation_folder, exist_ok=True)

        clear_folder(self.folder)
        clear_folder(self.last_frames_folder)

        # Receptors setup
        receptor_z = []
        for rp in range(0, 64):
            receptor_z.append((6.0e-4 * rp) / simulation_config['dz'])

        self.number_of_receptors = np.int32(64)
        self.receptor_z = np.int32(np.int32(np.asarray(receptor_z)) + np.int32((simulation_config['grid_size_z']
                                                                                - receptor_z[-1]) / 2))
        self.receptor_x = np.array([2000 for _ in range(64)], dtype=np.int32)

        # Reflectors setup
        self.number_of_reflectors = np.int32(5)
        self.reflector_z = self.receptor_z[np.asarray([0, 10, 32, 54, 63])]
        self.reflector_x = np.int32(np.asarray([4.139e-2, 4.592e-2, 5.796e-2, 6.995e-2, 7.500e-2]) / self.dx)

        # raw_b_scan = np.load('./panther/teste_3/ascan_data.npy')[:, 0, :, 0].transpose()
        raw_b_scan = np.load('./panther/teste_3/ascan_data.npy')[:, 32, :, 0].transpose()
        # raw_b_scan = np.load('./panther/teste_3/ascan_data.npy')[:, 54, :, 0].transpose()

        # Mic 0:
        # raw_b_scan[:, :1100] = np.float32(0)
        # raw_b_scan[:, 1645:] = np.float32(0)
        # # Mic 32:
        raw_b_scan[:29, 1800:] = np.float32(0)
        raw_b_scan[:, 2740:] = np.float32(0)
        # # Mic 54:
        # raw_b_scan[:50, 2300:] = np.float32(0)
        # raw_b_scan[:, 2690:] = np.float32(0)

        # plot_imshow(np.abs(raw_b_scan))

        zeros = np.zeros((64, 2000))
        raw_b_scan = np.hstack((zeros, raw_b_scan))

        simulation_b_scan = raw_b_scan[:, :]

        # plot_imshow(simulation_b_scan)

        self.tr_total_time = np.int32(len(simulation_b_scan[0]))
        np.save(f'{self.folder}/tr_total_time.npy', self.tr_total_time)
        print(f'Total time (TR): {self.tr_total_time}')

        self.reversed_pressure = []
        # Recorded pressure on receptors
        for i in range(self.number_of_receptors):
            self.reversed_pressure.append(np.array(np.flip(simulation_b_scan[i]), dtype=np.float32))

        self.info_int = np.array(
            [
                self.grid_size_z,
                self.grid_size_x,
                self.number_of_receptors,
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

        aux_string = ''
        for i in range(self.number_of_receptors):
            aux_string += f'''@group(0) @binding({i + 8})
var<storage,read> reversed_pressure_{i}: array<f32>;\n\n'''

        shader_string = shader_string.replace('//REVERSED_PRESSURE_BINDINGS', aux_string)

        aux_string = ''
        for i in range(self.number_of_receptors):
            aux_string += f'''if (receptor_idx == {i})
            {{
                p_future[zx(z, x)] += reversed_pressure_{i}[infoI32.i];
            }}\n'''

        shader_string = shader_string.replace('//REVERSED_PRESSURE_SIM', aux_string)

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

        scatter_kwargs = {
            'number_of_reflectors': self.number_of_reflectors,
            'reflector_z': self.reflector_z,
            'reflector_x': self.reflector_x + np.int32(2000),
            'number_of_receptors': self.number_of_receptors,
            'receptor_z': self.receptor_z,
            'receptor_x': self.receptor_x,
        }

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
                        scatter_kwargs=scatter_kwargs,
                        plt_kwargs=plt_kwargs,
                        plt_grid=True,
                        plt_colorbar=True,
                    )

            if i % 100 == 0:
                print(f'Time Reversal - i={i}')

        print('Time Reversal finished.')

        if create_animation:
            create_ffmpeg_animation(self.animation_folder, 'tr.mp4', self.tr_total_time, self.animation_step)
