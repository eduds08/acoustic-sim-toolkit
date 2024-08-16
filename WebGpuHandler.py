import wgpu
from webgpu_utils import read_shader_bindings


class WebGpuHandler:
    def __init__(self, *workgroup_sizes):
        self.shader_module = None
        self.pipeline_layout = None
        self.bind_groups = None

        self.ws = []

        for dim in workgroup_sizes:
            for i in range(15, 0, -1):
                if (dim % i) == 0:
                    self.ws.append(i)
                    break

        # GPU device
        self.device = wgpu.utils.get_default_device()

    def create_compute_pipeline(self, entry_point):
        """
        Creates a compute pipeline.

        Arguments:
            entry_point (str): @compute function's name declared in wgsl file.

        Returns:
            GPUComputePipeline.
        """
        return self.device.create_compute_pipeline(
            layout=self.pipeline_layout,
            compute={"module": self.shader_module, "entry_point": entry_point}
        )

    def create_buffers(self, data, shader_lines):
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

                buffers[f'b{binding}'] = self.create_buffer(data[data_and_binding_type[0]], binding,
                                                            buffer_binding_type,
                                                            bind_groups_layouts_entries[f'{group}'],
                                                            bind_groups_entries[f'{group}'])

        bind_groups_layouts_entries = dict(sorted(bind_groups_layouts_entries.items()))
        bind_groups_entries = dict(sorted(bind_groups_entries.items()))

        bind_groups_layouts_entries_list = [v for k, v in bind_groups_layouts_entries.items()]
        bind_groups_entries_list = [v for k, v in bind_groups_entries.items()]

        self.create_pipeline_layout(bind_groups_layouts_entries_list, bind_groups_entries_list)

        return buffers

    def create_buffer(
            self,
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
        new_buffer = self.device.create_buffer_with_data(data=data,
                                                         usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

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

    def create_pipeline_layout(self, bind_groups_layouts_entries: list, bind_groups_entries: list):
        """
        Creates bind_group_layouts and bind_groups from the arguments to create a pipeline_layout. Sets the class'
        pipeline layout and bind_groups.

        Arguments:
            bind_groups_layouts_entries (list): An element of this list is a dict with parameters that will help
                                                to build the bind_group_layout.
            bind_groups_entries (list): An element of this list is a dict with parameters that will help
                                        to build the bind_group.
        """
        bind_groups_layouts = []
        for bind_group_layout_entries in bind_groups_layouts_entries:
            bind_groups_layouts.append(self.device.create_bind_group_layout(entries=bind_group_layout_entries))

        self.pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=bind_groups_layouts)

        bind_groups = []
        for index, bind_group_entries in enumerate(bind_groups_entries):
            bind_groups.append(self.device.create_bind_group(layout=bind_groups_layouts[index],
                                                             entries=bind_group_entries))

        self.bind_groups = bind_groups
