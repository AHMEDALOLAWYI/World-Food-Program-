import numpy as np
from PIL import Image
import os
from pathlib import Path

DATA_PATH = 'lbl_data/'
TARGET_PATH = 'lbl_data_resized/'


def center_crop(img, new_width=64, new_height=None):
    width = img.width
    height = img.height

    if new_height is None:
        new_height = new_width

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    center_cropped_img = img.crop((left, top, right, bottom))

    assert ((center_cropped_img.height == new_height) and (center_cropped_img.width == new_width))

    return center_cropped_img


def crop_on_img(filename, path_target, size=41):
    img = Image.open(filename)
    img = center_crop(img, size, size)
    print(path_target + filename.stem + '.png')
    img.save(path_target + filename.stem + '.png')


if __name__ == '__main__':
    path_source = DATA_PATH
    # Target path of resized data
    path_target = TARGET_PATH

    if not os.path.exists(path_target):
        os.mkdir(path_target)

    for r, d, f in os.walk(path_source):
        if r != path_source:
            break
        print(f'In directory {r}: {len(f)} files')

        for file in f:
            try:
                # print(f'working on img {path_source+file}')
                image = Path(path_source + file)
                crop_on_img(image, path_target, 512)
            except:
                print(f'Exception occurred for {image}')
