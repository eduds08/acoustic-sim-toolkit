import numpy as np
import matplotlib.pyplot as plt
import wgpu
from webgpu_utils import read_shader_bindings

pipeline_layout = None
bind_groups = None

# GPU device
device = wgpu.utils.get_default_device()


def create_compute_pipeline(entry_point):
    """
    Creates a compute pipeline.

    Arguments:
        entry_point (str): @compute function's name declared in wgsl file.

    Returns:
        GPUComputePipeline.
    """
    return device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader_module, "entry_point": entry_point}
    )


def create_buffers(data, shader_lines):
    """
    Creates a dictionary containing all created buffers.

    Arguments:
        data (dict): Dictionary containing any object supporting the Python buffer protocol.
                     It's the data that will be passed to bindings on gpu.
        shader_lines (list): List of strings where each element is a line of a wgsl file.

    Returns:
        dict: Dictionary containing all created buffers. The key is a string named as 'b0', 'b1', etc...
              where 'b0' is @binding(0), for example. The value is a GPUBuffer.
    """
    buffers = dict()
    bind_groups_layouts_entries = dict()
    bind_groups_entries = dict()

    shader_bindings = read_shader_bindings(shader_lines)

    for group, binding_list in shader_bindings.items():
        for binding, data_and_binding_type in binding_list.items():
            buffer_binding_type = wgpu.BufferBindingType.read_only_storage if data_and_binding_type[1] == 'read' \
                else wgpu.BufferBindingType.storage

            if f'{group}' not in bind_groups_layouts_entries:
                bind_groups_layouts_entries[f'{group}'] = list()

            if f'{group}' not in bind_groups_entries:
                bind_groups_entries[f'{group}'] = list()

            buffers[f'b{binding}'] = create_buffer(data[data_and_binding_type[0]], binding,
                                                        buffer_binding_type,
                                                        bind_groups_layouts_entries[f'{group}'],
                                                        bind_groups_entries[f'{group}'])

    bind_groups_layouts_entries = dict(sorted(bind_groups_layouts_entries.items()))
    bind_groups_entries = dict(sorted(bind_groups_entries.items()))

    bind_groups_layouts_entries_list = [v for k, v in bind_groups_layouts_entries.items()]
    bind_groups_entries_list = [v for k, v in bind_groups_entries.items()]

    create_pipeline_layout(bind_groups_layouts_entries_list, bind_groups_entries_list)

    return buffers


def create_buffer(
        data,
        binding_number,
        buffer_binding_type,
        bind_groups_layouts_entries: list,
        bind_groups_entries: list
):
    """
    Creates a buffer using create_buffer_with_data() and also appends a dictionary with the passed arguments into
    bind_groups_layouts_entries and into bind_groups_entries. (Those two lists are passed by reference).

    Arguments:
        data (dict): Dictionary containing any object supporting the Python buffer protocol.
                     It's the data that will be passed to bindings on gpu.
        binding_number (int): The number 'x' specified in @binding(x).
        buffer_binding_type (enum): WebGPU binding type (read, read_only, etc...).
        bind_groups_layouts_entries (list): Passed by reference. An element of this list is a dict with parameters
                                            that will help to build the bind_group_layout.
        bind_groups_entries (list): Passed by reference. An element of this list is a dict with parameters that will
                                    help to build the bind_group.

    Returns:
        GPUBuffer object: Buffer created with create_buffer_with_data().
    """
    print(binding_number)
    new_buffer = device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

    bind_groups_layouts_entries.append({
        'binding': binding_number,
        'visibility': wgpu.ShaderStage.COMPUTE,
        'buffer': {
            "type": buffer_binding_type,
        }
    })

    bind_groups_entries.append({
        "binding": binding_number,
        "resource": {
            "buffer": new_buffer,
            "offset": 0,
            "size": new_buffer.size,
        }
    })

    return new_buffer


def create_pipeline_layout(bind_groups_layouts_entries: list, bind_groups_entries: list):
    """
    Creates bind_group_layouts and bind_groups from the arguments to create a pipeline_layout. Sets the class'
    pipeline layout and bind_groups.

    Arguments:
        bind_groups_layouts_entries (list): An element of this list is a dict with parameters that will help
                                            to build the bind_group_layout.
        bind_groups_entries (list): An element of this list is a dict with parameters that will help
                                    to build the bind_group.
    """
    global bind_groups, pipeline_layout

    bind_groups_layouts = []
    for bind_group_layout_entries in bind_groups_layouts_entries:
        bind_groups_layouts.append(device.create_bind_group_layout(entries=bind_group_layout_entries))

    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=bind_groups_layouts)

    bind_groups = []
    for index, bind_group_entries in enumerate(bind_groups_entries):
        bind_groups.append(device.create_bind_group(layout=bind_groups_layouts[index],
                                                         entries=bind_group_entries))

    bind_groups = bind_groups


fmc = np.load('teste4_results/ascan_data.npy')[:, :, :, 0]
fmc = np.array(fmc[:1700, :, :], dtype=np.float32)

image = np.zeros_like((fmc[:, :, 0]), dtype=np.float32)

wsReceptor = None
for i in range(15, 0, -1):
    if (len(fmc[0, 0, :]) % i) == 0:
        wsReceptor = i
        break

wsDepth = None
for i in range(15, 0, -1):
    if (len(image[:, 0]) % i) == 0:
        wsDepth = i
        break

wsLength = None
for i in range(15, 0, -1):
    if (len(image[0, :]) % i) == 0:
        wsLength = i
        break

shader_file = open('./tfm.wgsl')
shader_string = (shader_file.read()
                 .replace('wsReceptor', f'{wsReceptor}')
                 .replace('wsDepth', f'{wsDepth}')
                 .replace('wsLength', f'{wsLength}'))
shader_file.close()

shader_module = device.create_shader_module(code=shader_string)

time = np.float32(np.load('teste4_results/time_grid.npy') * 1e-6)
dx = np.float32(0.6e-3)
time_sample = np.float32((time[1] - time[0]).item())
acoustic_speed = np.float32(6420.)
delays = np.zeros_like(fmc, dtype=np.int32)

depth_length = np.int32(len(image[:, 0]))
transducer_length = np.int32(len(image[0, :]))

wgsl_data = {
    'time': time,
    'dx': dx,
    'time_sample': time_sample,
    'acoustic_speed': acoustic_speed,
    'delays': delays,
    'image': image,
    'fmc': fmc,
    'depth_length': depth_length,
    'transducer_length': transducer_length,
    'gate_start_frames': np.float32(500),
}

shader_lines = list(shader_string.split('\n'))
buffers = create_buffers(wgsl_data, shader_lines)

compute_create_delays = create_compute_pipeline("create_delays")
compute_sim_tfm = create_compute_pipeline("sim_tfm")

command_encoder = device.create_command_encoder()
compute_pass = command_encoder.begin_compute_pass()

for index, bind_group in enumerate(bind_groups):
    compute_pass.set_bind_group(index, bind_group, [], 0, 999999)

compute_pass.set_pipeline(compute_create_delays)
compute_pass.dispatch_workgroups(transducer_length // wsReceptor, depth_length // wsDepth, transducer_length // wsLength)

compute_pass.set_pipeline(compute_sim_tfm)
compute_pass.dispatch_workgroups(depth_length // wsDepth, transducer_length // wsLength)

compute_pass.end()
device.queue.submit([command_encoder.finish()])

""" READ BUFFERS """
image = (np.asarray(device.queue.read_buffer(buffers['b5']).cast("f"))
         .reshape(transducer_length, depth_length).transpose())

plt.figure()
plt.imshow(np.abs(image), aspect='auto')
plt.colorbar()
plt.show()
