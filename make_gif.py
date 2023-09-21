# taken from CS335, Fall'23, Lab6, Q1 make_gif.py
import os
import imageio

def make_directory_structure():
    os.makedirs('./animations', exist_ok=True)

def make_gif(name):
    make_directory_structure()
    png_dir = f'images/{name}'
    images = []
    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            for i in range(10): images.append(imageio.imread(file_path))

    # Make it pause at the end so that the viewers can ponder
    for _ in range(20):
        images.append(imageio.imread(file_path))

    imageio.mimsave(f'animations/{name}.gif', images)