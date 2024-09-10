import numpy as np
import os
from WebGPUConfig import WebGPUConfig
from plt_utils import save_imshow
from os_utils import clear_folder, create_ffmpeg_animation


class TimeReversal(WebGPUConfig):
    def __init__(self, simulation_config, tr_config):
        super().__init__(**simulation_config)

        tr_config = {**tr_config}

        self.shader_file = './time_reversal.wgsl'

        self.recorded_pressure_bscan = tr_config['recorded_pressure_bscan']

        self.setup_folders()

        recs_dist = tr_config['recs_dist']

        # Receptors setup
        self.number_of_receptors = len(self.recorded_pressure_bscan[:, 0])
        self.receptor_z = []
        for rp in range(self.number_of_receptors):
            self.receptor_z.append((recs_dist * rp) / self.dz)
        self.receptor_z = (np.int32(np.asarray(self.receptor_z))
                           + np.int32((self.grid_size_z - self.receptor_z[-1]) / 2))
        self.receptor_x = np.full(self.number_of_receptors, 2, dtype=np.int32)

        np.save(f'{self.folder}/source_z.npy', self.receptor_z[tr_config['emitter_index']])
        np.save(f'{self.folder}/source_x.npy', self.receptor_x[tr_config['emitter_index']])

        # Reflectors setup
        self.number_of_reflectors = np.int32(1)
        self.reflector_z = np.array([self.receptor_z[tr_config['emitter_index']]], dtype=np.int32)
        self.reflector_x = np.array([tr_config['distance_from_reflector'] / self.dx], dtype=np.int32)

        np.save(f'{self.folder}/number_of_reflectors.npy', self.number_of_reflectors)
        np.save(f'{self.folder}/reflector_z.npy', self.reflector_z)
        np.save(f'{self.folder}/reflector_x.npy', self.reflector_x)

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

        for i in range(self.number_of_receptors):
            recorded_pressure = self.recorded_pressure_bscan[i][self.min_time:self.max_time]

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

        scatter_kwargs = {
            'number_of_reflectors': self.number_of_reflectors,
            'reflector_z': self.reflector_z,
            'reflector_x': self.reflector_x,
            'number_of_receptors': self.number_of_receptors,
            'receptor_z': self.receptor_z,
            'receptor_x': self.receptor_x,
        }

        l2_norm = np.zeros_like(self.p_future)

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

            l2_norm += np.square(self.p_future)

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

        l2_norm = np.sqrt(l2_norm)

        np.save(f'{self.folder}/l2_norm.npy', l2_norm)

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
