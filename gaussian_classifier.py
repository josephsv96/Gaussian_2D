import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical


class SegmentaionMap:
    def __init__(self, annot_arr):
        self.annot_arr = annot_arr

    def arr_to_categorical(self):
        annot_arr_cat = to_categorical(
            self.annot_arr, num_classes=None, dtype='float32')

        return annot_arr_cat

    def get_gaussian(self):
        annot_arr = self.arr_to_categorical()
        return annot_arr

    def activation_density(self, image_arr):
        activation_densities = np.zeros(image_arr.shape[-1])
        total_pixels = image_arr.shape[0] * image_arr.shape[1]

        for i in range(image_arr.shape[-1]):
            image = image_arr[:, :, i]
            activated_pixels = image[image > 0]
            activation_densities[i] = activated_pixels.shape[0] / total_pixels

        return activation_densities

    def get_density(self):
        annot_arr = self.arr_to_categorical()
        batch_size = annot_arr.shape[0]
        densities_list = np.zeros([batch_size, annot_arr.shape[-1]])

        for i in range(batch_size):
            densities_list[i, :] = self.activation_density(
                annot_arr[i, :, :, :])

        return densities_list

    def imshow_prediciton(self):
        densities_list = self.get_density()

        plt.figure(figsize=(15, 10))
        j = 0
        num_of_images = self.annot_arr.shape[0] + 1
        for i in range(self.annot_arr.shape[0]):
            plt.subplot(1 * num_of_images, 2, j+1)
            plt.imshow(self.annot_arr[i, :, :])
            # print(densities_list[i, 1:] * 100)

            plt.subplot(1 * num_of_images, 2, j+2)
            plt.plot(densities_list[i, 1:] * 100)
            plt.xlabel('Class')

            j = j + 2
