import numpy as np
from pathlib import Path
from img_bmp_utils import read_cmp, save_cmp_npy


class CmapSplitter:
    def __init__(self, annot_arr):
        self.annot_arr = annot_arr

    def get_box_values(self):

    def split_to_4(self, width, height, x_offset, y_offset):
        split_annot_arr_shape = (self.annot_arr.shape[0]*4, width, height)
        split_annot_arr = np.zeros(split_annot_arr_shape)


        return split_annot_arr


def main():
    DATA_PATH = Path(input("Enter annotation directory: "))

    cmp_files = list(DATA_PATH.glob('**/*.cmp'))

    width = 256
    height = 256
    # channels = 3

    # Read the bmp, cmp files
    label_set = read_cmp(cmp_files, width, height)

    # Saving the images to as NPY files
    save_cmp_npy(DATA_PATH, label_set)


if __name__ == "__main__":
    main()
