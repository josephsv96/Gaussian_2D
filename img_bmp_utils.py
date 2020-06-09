# import json
import numpy as np
import matplotlib.pyplot as plt

import cv2
from pathlib import Path


def read_bmp(img_files, width, height, channels):
    img_num = len(img_files)
    image_set = np.zeros([img_num, width, height, channels], dtype=np.uint8)

    for i, file in enumerate(img_files):
        src = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
        src = cv2.resize(src, (width, height))
        image_set[i, :, :, 0] = src[:, :, 2]
        image_set[i, :, :, 1] = src[:, :, 1]
        image_set[i, :, :, 2] = src[:, :, 0]

    image_set = image_set.reshape(img_num, width, height, channels)

    return image_set


def read_cmp(segmap_files, width, height):
    segmap_num = len(segmap_files)
    # Read the cmp files
    label_set = np.zeros([segmap_num, width, height], dtype=np.int8)

    for i, file in enumerate(segmap_files):

        # open binary file
        file = open(file, "rb")

        # get file size
        file.seek(0, 2)    # set pointer to end of file
        nb_bytes = file.tell()
        file.seek(0, 0)    # set pointer to begin of file
        buf = file.read(nb_bytes)
        file.close()

        # convert from byte stream to numpy array
        segmap = np.asarray(list(buf), dtype=np.byte)
        current_dimension = int(segmap.shape[0]**0.5)

        segmap = segmap.reshape([current_dimension, current_dimension])

        segmap = cv2.resize(np.array(segmap, dtype='uint8'),
                            (width, height),  interpolation=cv2.INTER_LINEAR)
        # segmap = np.resize(segmap, [width, height])
        # segmap = segmap.reshape([width, height])

        label_set[i, :, :] = segmap
        print("Saving:", i+1, '/', segmap_num)

    return label_set


def read_bmp_cmp(img_files, cmap_files, width, height, channels):
    image_arr = read_bmp(img_files, width, height, channels)
    annot_arr = read_cmp(cmap_files, width, height)

    return image_arr, annot_arr


def save_npy(CWD_PATH, image_arr, label_arr):
    # Destination directory of dataset
    OUTPUT_PATH = CWD_PATH / 'dataset'
    if OUTPUT_PATH.is_dir():
        print("Dataset folder exists; overwriting existing files in folder")
    else:
        print("Dataset folder created")
        OUTPUT_PATH.mkdir()

    # Saving as .npy files
    np.save(OUTPUT_PATH / 'dataset_images', image_arr)
    np.save(OUTPUT_PATH / 'dataset_labels', label_arr)


def read_npy(CWD_PATH):
    # Testing the output files
    image_arr = np.load(CWD_PATH/'dataset_images.npy')
    annot_arr = np.load(CWD_PATH/'dataset_labels.npy')

    print("Image array:", image_arr.shape)
    print("Annotation array:", annot_arr.shape)

    return image_arr, annot_arr


def main():
    """read BMP and CMP files from DATA_PATH and saves NPY arrays for each
    """
    # Future: use arg parese here
    DATA_PATH = Path(input("Enter image directory: "))

    img_files = list(DATA_PATH.glob('**/*.bmp'))
    cmp_files = list(DATA_PATH.glob('**/*.cmp'))
    # img_num = len(img_files)    # number of image files
    # cmp_num = len(cmp_files)    # number of annotation files

    """exp_num = int(img_num / cmp_num)

    if img_num % cmp_num != 0:
        print("Missing images or annotations")
    else:
        print("Found " + str(img_num) + " BMP files")
        print("Found " + str(cmp_num) + " CMP files")
        print("Exposures: " + str(exp_num))
        print("Batch size: " + str(cmp_num))
    """

    # Needed output shape of image batch
    # => (batch_len, exposure_level, width, height, 3)
    # Needed output shape of annotation batch
    # => (batch_len)
    # !! Future, this should be a JSON config
    width = 128
    height = 128
    channels = 3

    # Read the bmp, cmp files
    image_set, label_set = read_bmp_cmp(
        img_files, cmp_files, width, height, channels)

    # Saving the images to as NPY files
    save_npy(DATA_PATH, image_set, label_set)

    i = 0
    # plt.imshow(image_set[i][0])
    # plt.show()

    plt.imshow(label_set[i])
    plt.show()


if __name__ == "__main__":
    main()
