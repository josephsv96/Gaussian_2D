from img_bmp_utils import read_cmp, save_cmp_npy
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np


class CmapSplitter:
    def __init__(self, annot_arr, output_shape=[128, 128]):
        self.annot_arr = annot_arr
        self.output_shape = output_shape

    def get_config(self):
        annot_arr = self.annot_arr
        y_offset_arr = [0, 0, 105, 100]

        height_arr = [110, 100,
                      annot_arr.shape[1] - y_offset_arr[2],
                      annot_arr.shape[1] - y_offset_arr[3]]

        x_offset_arr = [0, 115, 0, 120]
        width_arr = [110,
                     annot_arr.shape[2] - x_offset_arr[2],
                     120,
                     annot_arr.shape[2] - x_offset_arr[3]]

        split_params = [y_offset_arr, height_arr, x_offset_arr, width_arr]

        return split_params

    def get_box_values(self):
        return

    def split_to_4(self, split_config=[]):

        annot_arr = self.annot_arr
        out_height = self.output_shape[0]
        out_width = self.output_shape[1]

        if len(split_config) == 0:
            print("Using in-built config")
            split_config = self.get_config()

        y_offset_arr = split_config[0]
        height_arr = split_config[1]
        x_offset_arr = split_config[2]
        width_arr = split_config[3]

        split_annot_arr_shape = (annot_arr.shape[0]*4, out_height, out_width)
        split_annot_arr = np.zeros(split_annot_arr_shape)

        count = 0
        for annot in annot_arr:
            for i in range(4):
                height_ix = [y_offset_arr[i], y_offset_arr[i] + height_arr[i]]
                width_ix = [x_offset_arr[i], x_offset_arr[i] + width_arr[i]]

                split_img = annot[height_ix[0]:height_ix[1],
                                  width_ix[0]:width_ix[1], ]
                buffer_img = np.zeros(
                    [split_img.shape[0], split_img.shape[1], 1])
                buffer_img[:, :, 0] = split_img

                split_annot_arr[count, :, :] = cv2.resize(
                    buffer_img[:, :, 0], dsize=(out_height, out_width))

                count += 1

        return split_annot_arr

    # def show_split(self, index=0):
    #     plt.imshow(annot_arr[index])
    #     plt.grid('on')

    #     j = index * 4
    #     plt.figure(figsize=(6, 6))
    #     for k in range(4):
    #         plt.subplot(2, 2, k+1)
    #         plt.imshow(split_annot_arr[j+k, :])
    #         plt.grid('on')
    #         plt.xticks([32, 64, 96])
    #         plt.yticks([32, 64, 96])
    #         plt.colorbar()

    #     plt.show()


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
