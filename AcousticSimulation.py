import numpy as np
import os
from WebGPUConfig import WebGPUConfig
from plt_utils import save_imshow
from os_utils import clear_folder, create_ffmpeg_animation
import matplotlib.pyplot as plt
import matplotlib
import findiff
import pyqtgraph as pg
from pyqtgraph.widgets.RawImageWidget import RawImageWidget, RawImageGLWidget

class AcousticSimulation(WebGPUConfig):
    def __init__(self, simulation_config, acoustic_config):
        super().__init__(**simulation_config)

        acoustic_config = {**acoustic_config}

        self.shader_file = './acoustic_simulation.wgsl'

        self.setup_folders()

        # Source position
        self.source_z = np.int32(acoustic_config['source_z'])
        self.source_x = np.int32(acoustic_config['source_x'])
        np.save(f'{self.source_folder}/source_z.npy', self.source_z)
        np.save(f'{self.source_folder}/source_x.npy', self.source_x)

        # Reflectors setup
        self.reflector_z, self.reflector_x = np.int32(np.where(self.c == 0))
        self.number_of_reflectors = len(self.reflector_z)
        np.save(f'{self.reflectors_folder}/number_of_reflectors.npy', self.number_of_reflectors)
        np.save(f'{self.reflectors_folder}/reflector_z.npy', self.reflector_z)
        np.save(f'{self.reflectors_folder}/reflector_x.npy', self.reflector_x)

        # Receptors setup
        self.number_of_receptors = np.int32(acoustic_config['number_of_receptors'])
        self.receptor_z = np.array(acoustic_config['receptor_z'], dtype=np.int32)
        self.receptor_x = np.array(acoustic_config['receptor_x'], dtype=np.int32)
        np.save(f'{self.receptors_folder}/number_of_receptors.npy', self.number_of_receptors)
        np.save(f'{self.receptors_folder}/receptor_z.npy', self.receptor_z)
        np.save(f'{self.receptors_folder}/receptor_x.npy', self.receptor_x)

        self.recs = []
        for _ in range(self.number_of_receptors):
            self.recs.append(np.array([0 for _ in range(self.total_time)], dtype=np.float32))

        # Source
        time_arr = np.linspace(0, self.total_time * self.dt, self.total_time, dtype=np.float32)
        f0 = 10
        t0 = 2 / f0
        self.source = np.array(-8. * (time_arr - t0) * f0 * (np.exp(-1. * (time_arr - t0) ** 2 * (f0 * 4) ** 2)),
                               dtype=np.float32)

        np.save(f'{self.source_folder}/source.npy', self.source)

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
            ],
            dtype=np.float32
        )

    def run(self, create_animation: bool, **plt_kwargs):
        # shader_file = open(self.shader_file)
        # shader_string = shader_file.read().replace('wsz', f'{self.wsz}').replace('wsx', f'{self.wsx}')
        # shader_file.close()
        #
        # self.shader_module = self.device.create_shader_module(code=shader_string)

        wgsl_data = {
            'infoI32': self.info_int,
            'infoF32': self.info_float,
            'source': self.source,
            'p_future': self.p_future,
            'p_present': self.p_present,
            'p_past': self.p_past,
            'lap': self.lap,
            'c': self.c,
        }

        # shader_lines = list(shader_string.split('\n'))
        # buffers = self.create_buffers(wgsl_data, shader_lines)
        #
        # compute_lap = self.create_compute_pipeline("laplacian_5_operator")
        # compute_sim = self.create_compute_pipeline("sim")
        # compute_incr = self.create_compute_pipeline("incr_time")

        def derivative(dim, u, boundary_factor):
            accuracy = 8
            offsets = (np.concatenate((np.arange(-accuracy // 2, 0), np.arange(accuracy // 2))) + 0.5).astype(
                np.float32)
            coeff_diff = findiff.coefficients(deriv=1, offsets=list(offsets))["coefficients"][
                         round(accuracy / 2):].astype(np.float32)

            shifted_u = u.swapaxes(0, dim)
            derivative_result = np.zeros((shifted_u.shape[0] + 1, shifted_u.shape[1]), dtype=np.float32)

            for idx, coeff in enumerate(coeff_diff):
                derivative_result[:-idx - 1, ...] += coeff * shifted_u[idx:, ...]
                derivative_result[idx + 1:, ...] -= coeff * shifted_u[:shifted_u.shape[0] - idx, ...]

            return derivative_result[boundary_factor:derivative_result.shape[0] + boundary_factor - 1, ...].swapaxes(0,
                                                                                                                     dim)

        absorption_layer_size = 40
        damping_coefficient = 3e6

        x, z = np.meshgrid(np.arange(self.grid_size_x, dtype=np.float32), np.arange(self.grid_size_z, dtype=np.float32))

        # Aqui escolhemos as bordas que queremos absorver
        is_x_absorption = (x > self.grid_size_x - absorption_layer_size) | (x < absorption_layer_size)
        is_z_absorption = (z > self.grid_size_z - absorption_layer_size) | (z < absorption_layer_size)

        absorption_coefficient = np.exp(
            -(damping_coefficient * (np.arange(absorption_layer_size) / absorption_layer_size) ** 2) * self.dt).astype(
            np.float32)

        psi_x = np.zeros_like(is_x_absorption, dtype=np.float32)
        psi_z = np.zeros_like(is_z_absorption, dtype=np.float32)
        phi_x = np.zeros_like(is_x_absorption, dtype=np.float32)
        phi_z = np.zeros_like(is_z_absorption, dtype=np.float32)

        absorption_x = np.ones((self.grid_size_z, self.grid_size_x), dtype=np.float32)
        absorption_z = np.ones((self.grid_size_z, self.grid_size_x), dtype=np.float32)

        # Aqui também escolhems as bordas que queremos absorver
        absorption_x[:, :absorption_layer_size] = absorption_coefficient[::-1]
        absorption_x[:, -absorption_layer_size:] = absorption_coefficient
        absorption_z[:absorption_layer_size, :] = absorption_coefficient[:, np.newaxis][::-1]
        absorption_z[-absorption_layer_size:, :] = absorption_coefficient[:, np.newaxis]

        if create_animation:
            clear_folder(self.animation_folder)

        # # GUI (animação)
        # vminmax = 1e-4
        # vscale = 1
        # surface_format = pg.QtGui.QSurfaceFormat()
        # surface_format.setSwapInterval(0)
        # pg.QtGui.QSurfaceFormat.setDefaultFormat(surface_format)
        # app = pg.QtWidgets.QApplication([])
        # raw_image_widget = RawImageGLWidget()
        # raw_image_widget.setWindowFlags(pg.QtCore.Qt.WindowType.FramelessWindowHint)
        # raw_image_widget.resize(vscale * self.grid_size_x, vscale * self.grid_size_z)
        # raw_image_widget.show()
        # colormap = plt.get_cmap("bwr")
        # norm = matplotlib.colors.Normalize(vmin=-vminmax, vmax=vminmax)

        #SIMULATION#
        for i in range(self.total_time):
            z_diff_1 = derivative(dim=0, u=self.p_present, boundary_factor=1)
            x_diff_1 = derivative(dim=1, u=self.p_present, boundary_factor=1)

            phi_z[is_z_absorption] = absorption_z[is_z_absorption] * phi_z[is_z_absorption] + (
                        absorption_z[is_z_absorption] - 1) * z_diff_1[is_z_absorption]
            phi_x[is_x_absorption] = absorption_x[is_x_absorption] * phi_x[is_x_absorption] + (
                        absorption_x[is_x_absorption] - 1) * x_diff_1[is_x_absorption]

            z_diff_1[is_z_absorption] += phi_z[is_z_absorption]
            x_diff_1[is_x_absorption] += phi_x[is_x_absorption]

            z_diff_2 = derivative(dim=0, u=z_diff_1, boundary_factor=0)
            x_diff_2 = derivative(dim=1, u=x_diff_1, boundary_factor=0)

            psi_z[is_z_absorption] = absorption_z[is_z_absorption] * psi_z[is_z_absorption] + (
                        absorption_z[is_z_absorption] - 1) * z_diff_2[is_z_absorption]
            psi_x[is_x_absorption] = absorption_x[is_x_absorption] * psi_x[is_x_absorption] + (
                        absorption_x[is_x_absorption] - 1) * x_diff_2[is_x_absorption]

            z_diff_2[is_z_absorption] += psi_z[is_z_absorption]
            x_diff_2[is_x_absorption] += psi_x[is_x_absorption]

            z_diff_2 /= (self.dz ** 2)
            x_diff_2 /= (self.dx ** 2)

            self.p_future = (self.c ** 2) * (z_diff_2 + x_diff_2) * (self.dt ** 2)

            self.p_future += 2 * self.p_present - self.p_past

            self.p_past = self.p_present
            self.p_present = self.p_future

            self.p_future[self.source_z, self.source_x] += self.source[i]

            # Record pressure on recepetors
            for j in range(self.number_of_receptors):
                self.recs[j][i] = self.p_future[self.receptor_z[j], self.receptor_x[j]]

            if i % self.animation_step == 0:
                if create_animation:
                    save_imshow(
                        data=self.p_future,
                        title=f'Acoustic Simulation',
                        path=f'{self.animation_folder}/plot_{i}.png',
                        scatter_kwargs={
                            'number_of_reflectors': self.number_of_reflectors,
                            'reflector_z': self.reflector_z,
                            'reflector_x': self.reflector_x,
                            'number_of_receptors': self.number_of_receptors,
                            'receptor_z': self.receptor_z,
                            'receptor_x': self.receptor_x,
                        },
                        **plt_kwargs,
                    )

            if i % 100 == 0:
                print(f'Acoustic Simulation - i={i}')

            # # Atualiza a GUI
            # if not i % 3:
            #     raw_image_widget.setImage(colormap(norm(self.p_future.T)), levels=[0, 1])
            #     app.processEvents()
            #     plt.pause(1e-12)

        print('Acoustic Simulation finished.')

        # Save receptors' recorded pressure
        for j in range(self.number_of_receptors):
            np.save(f'{self.recorded_pressure_folder}/receptor_{j}.npy', self.recs[j])

        if create_animation:
            create_ffmpeg_animation(self.animation_folder, 'ac_sim.mp4', self.total_time, self.animation_step)


    def setup_folders(self):
        self.folder = './AcousticSimulation'
        self.source_folder = f'{self.folder}/source_setup'
        self.receptors_folder = f'{self.folder}/receptors_setup'
        self.recorded_pressure_folder = f'{self.folder}/recorded_pressure'
        self.reflectors_folder = f'{self.folder}/reflectors_setup'
        self.animation_folder = f'{self.folder}/animation'

        os.makedirs(self.folder, exist_ok=True)
        os.makedirs(self.source_folder, exist_ok=True)
        os.makedirs(self.receptors_folder, exist_ok=True)
        os.makedirs(self.recorded_pressure_folder, exist_ok=True)
        os.makedirs(self.reflectors_folder, exist_ok=True)
        os.makedirs(self.animation_folder, exist_ok=True)

        clear_folder(self.source_folder)
        clear_folder(self.receptors_folder)
        clear_folder(self.recorded_pressure_folder)
        clear_folder(self.reflectors_folder)
