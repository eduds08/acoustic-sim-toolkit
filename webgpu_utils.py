def read_shader_bindings(shader_lines: list):
    """
    Creates a dictionary containing important information about bindings on wgsl file.

    Arguments:
        shader_lines (list): List of strings where each element is a line of a wgsl file.

    Returns:
        dict: a dictionary containing the group's number, binding's number, binding's type (read or read_only) and the
              binding's name.
    """
    shader_bindings = dict()

    last_group = None
    last_binding = None

    for line in shader_lines:
        if line.find(f'@group(') != -1:
            current_group = ''.join(line.split('@group(')[1].split(')')[0])
            current_binding = ''.join(line.split('@binding(')[1].split(')')[0])

            if f'{current_group}' not in shader_bindings:
                shader_bindings[f'{current_group}'] = dict()

            shader_bindings[f'{current_group}'][f'{current_binding}'] = []

            last_group = current_group
            last_binding = current_binding

        elif line.find('var<storage,') != -1:
            buffer_binding_type = ''.join(line.split('var<storage,')[1].split('>')[0])

            binding_name = ''.join(line.split('>')[1].split(':')[0])
            binding_name = binding_name.strip()

            shader_bindings[last_group][last_binding].append(binding_name)
            shader_bindings[last_group][last_binding].append(buffer_binding_type)

    return shader_bindings
