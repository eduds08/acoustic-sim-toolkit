import os


def clear_folder(folder):
    for file in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, file)):
            os.remove(f'{folder}/{file}')


def create_ffmpeg_animation(folder, file_name, total_time, step):
    parameters_file = f'{folder}/parameters_ffmpeg.txt'

    with open(parameters_file, 'w') as f:
        [f.write(f"file 'plot_{t}.png'\nduration 0.1\n") for t in range(0, total_time, step)]

    os.system(f'ffmpeg -f concat -safe 0 -i {parameters_file} -c:v libx264rgb -crf 0 '
              f'{folder}/{file_name}')
