import numpy as np
from WebGpuHandler import WebGpuHandler
from plt_utils import plot_imshow, save_imshow


class TFM:
    def __init__(self, test: list):
        self.test = test

        self.fmc = np.load(f'{self.test[0]}/ascan_data.npy')[:, :, :, 0]
        self.fmc = np.array(self.fmc[:, :, :], dtype=np.float32)

        # plot_imshow(np.abs(self.fmc[:, test[4], :]), 'Raw BScan', {}, aspect='auto')

        self.image = np.zeros_like((self.fmc[:, :, 0]), dtype=np.float32)
        self.time = np.float32(np.load(f'{self.test[0]}/time_grid.npy') * 1e-6)
        self.dx = np.float32(0.6e-3)
        self.sample_time = np.float32((self.time[1] - self.time[0]).item())
        self.acoustic_speed = np.float32(self.test[3])
        self.delays = np.zeros_like(self.fmc, dtype=np.int32)

        self.depth_length = np.int32(len(self.image[:, 0]))
        self.transducer_length = np.int32(len(self.image[0, :]))

        self.gpuHandler = WebGpuHandler(len(self.fmc[0, 0, :]), len(self.image[:, 0]), len(self.image[0, :]))

    def run(self):
        shader_file = open('./tfm.wgsl')
        shader_string = (shader_file.read()
                         .replace('wsReceptor', f'{self.gpuHandler.ws[0]}')
                         .replace('wsDepth', f'{self.gpuHandler.ws[1]}')
                         .replace('wsLength', f'{self.gpuHandler.ws[2]}'))
        shader_file.close()

        self.gpuHandler.shader_module = self.gpuHandler.device.create_shader_module(code=shader_string)

        # Add zeros, according to gate_start, to the beginning of array
        gate_start_value = np.float32(0)
        with open(f'{self.test[0]}/inspection_params.txt', 'r') as f:
            for line in f:
                if line.startswith('gate_start'):
                    gate_start_value = np.float32(line.split('=')[1].strip())
                    break
        padding_zeros = np.float32(gate_start_value / self.sample_time) * 1e-6

        wgsl_data = {
            'time': self.time,
            'dx': self.dx,
            'sample_time': self.sample_time,
            'acoustic_speed': self.acoustic_speed,
            'delays': self.delays,
            'image': self.image,
            'fmc': self.fmc,
            'depth_length': self.depth_length,
            'transducer_length': self.transducer_length,
            'gate_start_frames': padding_zeros,
        }

        shader_lines = list(shader_string.split('\n'))
        buffers = self.gpuHandler.create_buffers(wgsl_data, shader_lines)

        compute_create_delays = self.gpuHandler.create_compute_pipeline("create_delays")
        compute_sim_tfm = self.gpuHandler.create_compute_pipeline("sim_tfm")

        command_encoder = self.gpuHandler.device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()

        for index, bind_group in enumerate(self.gpuHandler.bind_groups):
            compute_pass.set_bind_group(index, bind_group, [], 0, 999999)

        compute_pass.set_pipeline(compute_create_delays)
        compute_pass.dispatch_workgroups(self.transducer_length // self.gpuHandler.ws[0],
                                         self.depth_length // self.gpuHandler.ws[1],
                                         self.transducer_length // self.gpuHandler.ws[2])

        compute_pass.set_pipeline(compute_sim_tfm)
        compute_pass.dispatch_workgroups(self.depth_length // self.gpuHandler.ws[1],
                                         self.transducer_length // self.gpuHandler.ws[2])

        compute_pass.end()
        self.gpuHandler.device.queue.submit([command_encoder.finish()])

        """ READ BUFFERS """
        image = (np.asarray(self.gpuHandler.device.queue.read_buffer(buffers['b5']).cast("f"))
                 .reshape(self.transducer_length, self.depth_length).transpose())

        if self.test[0] == './panther/teste3_results':
            plot_imshow(
                data=np.abs(image),
                title='TFM',
                scatter_kwargs={},
                aspect='auto'
            )
        else:
            # plot_imshow(
            #     data=np.abs(image[:int((self.test[2] / self.dx) * 2), :]),
            #     title='TFM',
            #     scatter_kwargs={},
            #     aspect='auto'
            # )
            plot_imshow(
                data=np.abs(image[:300]),
                title='TFM',
                scatter_kwargs={},
                aspect='auto'
            )
            save_imshow(
                data=np.abs(image[:300]),
                title='TFM',
                path='./tfm_teste6.png',
                scatter_kwargs={},
                aspect='auto'
            )
