import os


def clear_folder(folder):
    for file in os.listdir(folder):
        os.remove(f'{folder}/{file}')


def create_ffmpeg_animation(folder, file_name, total_time, step):
    parameters_file = f'{folder}/parameters_ffmpeg.txt'

    with open(parameters_file, 'w') as f:
        [f.write(f"file 'plot_{t}.png'\n") for t in range(0, total_time, step)]

    with open(parameters_file, 'a') as f:
        f.write(f"file 'plot_{total_time - 1}.png'\n")

    os.system(f'ffmpeg -f concat -safe 0 -i {parameters_file} -c:v copy '
              f'{folder}/{file_name}')
