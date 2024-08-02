import os


def clear_folder(folder):
    for file in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, file)):
            os.remove(f'{folder}/{file}')
