import numpy as np
import os
import matplotlib.pyplot as plt
from WebGPUConfig import WebGPUConfig
from plt_utils import save_imshow
from os_utils import clear_folder, create_ffmpeg_animation


class TimeReversal(WebGPUConfig):
    def __init__(self, simulation_config, tr_config):
        super().__init__(**simulation_config)

        tr_config = {**tr_config}

        self.shader_file = './time_reversal.wgsl'

        self.setup_folders()

        # Reflectors setup
        if len(os.listdir(f'{self.ac_reflectors_folder}')) != 0:
            self.number_of_reflectors = np.load(f'{self.ac_reflectors_folder}/number_of_reflectors.npy')
            self.reflector_z = np.load(f'{self.ac_reflectors_folder}/reflector_z.npy')
            self.reflector_x = np.load(f'{self.ac_reflectors_folder}/reflector_x.npy')

        # Receptors setup
        self.number_of_receptors = np.load(f'{self.ac_receptors_folder}/number_of_receptors.npy')
        self.receptor_z = np.load(f'{self.ac_receptors_folder}/receptor_z.npy')
        self.receptor_x = np.load(f'{self.ac_receptors_folder}/receptor_x.npy')

        # Slice
        self.min_time = np.int32(tr_config['min_time'])
        self.max_time = np.int32(tr_config['max_time'])
        self.padding_zeros = np.int32(tr_config['padding_zeros'])

        # Total time
        self.tr_total_time = np.int32(self.padding_zeros + (self.max_time - self.min_time))
        np.save(f'{self.folder}/tr_total_time.npy', self.tr_total_time)
        print(f'Total time (TR): {self.tr_total_time}')

        # Recorded pressure on receptors
        self.reversed_pressure = []

        source = np.load(f'{self.ac_sim_folder}/source_setup/source.npy')
        source_index = ~np.isclose(source, 0)

        for i in range(self.number_of_receptors):
            recorded_pressure = np.load(
                f'{self.ac_recorded_pressure_folder}/receptor_{i}.npy'
            )[self.min_time:self.max_time]

            # Cut the recorded source (when receptor on x=2)
            if self.receptor_x[0] == 2:
                recorded_pressure[~np.isclose(source_index, 0)] = np.float32(0)

            flipped_recorded_pressure = np.array(np.flip(recorded_pressure), dtype=np.float32)

            padded_recorded_pressure = np.pad(flipped_recorded_pressure, (0, self.padding_zeros), mode='constant')

            self.reversed_pressure.append(padded_recorded_pressure)

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
            ],
            dtype=np.float32
        )

    def run(self, create_animation: bool, **plt_kwargs):
        shader_file = open(self.shader_file)
        shader_string = shader_file.read().replace('wsz', f'{self.wsz}').replace('wsx', f'{self.wsx}')

        last_binding = 8

        aux_string = ''
        for i in range(self.number_of_receptors):
            aux_string += f'''@group(0) @binding({i + (last_binding + 1)})
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
            'c': self.c,
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

        if len(os.listdir(f'{self.ac_reflectors_folder}')) != 0:
            scatter_kwargs = {
                'number_of_reflectors': self.number_of_reflectors,
                'reflector_z': self.reflector_z,
                'reflector_x': self.reflector_x,
                'number_of_receptors': self.number_of_receptors,
                'receptor_z': self.receptor_z,
                'receptor_x': self.receptor_x,
            }
        else:
            scatter_kwargs = {
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

            # Save last 2 frames (for RTM)
            if i == self.tr_total_time - 1 or i == self.tr_total_time - 2:
                np.save(f'{self.last_frames_folder}/tr_{i}', self.p_future)

            if i % self.animation_step == 0:
                if create_animation:
                    save_imshow(
                        data=self.p_future,
                        title=f'Time Reversal',
                        path=f'{self.animation_folder}/plot_{i}.png',
                        scatter_kwargs=scatter_kwargs,
                        **plt_kwargs,
                    )

            if i % 100 == 0:
                print(f'Time Reversal - i={i}')

        print('Time Reversal finished.')

        if create_animation:
            create_ffmpeg_animation(self.animation_folder, 'tr.mp4', self.tr_total_time, self.animation_step)

    def setup_folders(self):
        self.folder = './TimeReversal'
        self.last_frames_folder = f'{self.folder}/last_frames'
        self.animation_folder = f'{self.folder}/animation'

        os.makedirs(self.folder, exist_ok=True)
        os.makedirs(self.last_frames_folder, exist_ok=True)
        os.makedirs(self.animation_folder, exist_ok=True)

        clear_folder(self.folder)
        clear_folder(self.last_frames_folder)

        self.ac_sim_folder = './AcousticSimulation'
        self.ac_receptors_folder = f'{self.ac_sim_folder}/receptors_setup'
        self.ac_recorded_pressure_folder = f'{self.ac_sim_folder}/recorded_pressure'
        self.ac_reflectors_folder = f'{self.ac_sim_folder}/reflectors_setup'
