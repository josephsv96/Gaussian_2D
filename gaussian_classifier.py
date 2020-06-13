import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from scipy.stats import norm


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
        if num_of_images > 10:
            num_of_images = 10
        for i in range(self.annot_arr.shape[0]):
            plt.subplot(1 * num_of_images, 2, j+1)
            plt.imshow(self.annot_arr[i, :, :])
            # print(densities_list[i, 1:] * 100)

            plt.subplot(1 * num_of_images, 2, j+2)
            plt.plot(densities_list[i, :] * 100)
            plt.xlabel('Class')

            j = j + 2

    def class_dist(self, low_threshold=0.15):
        densities_list = self.get_density()
        num_classes = densities_list.shape[-1]
        num_annots = densities_list.shape[0]

        # up_threshold = 0.45
        # Average of prediction over all classes
        class_dist = np.zeros([num_classes, num_annots, num_classes])
        for i in range(num_annots):
            # print(densities_list[i, :])
            class_index = np.where(densities_list[i, 1:] > low_threshold)[0][0]

            class_dist[class_index, i, 1:] = densities_list[i, 1:]

        class_dist = np.sum(class_dist, axis=1)

        return class_dist

    def show_class_dist(self):
        densities_list = self.class_dist()
        num_classes = densities_list.shape[0]
        class_colors = ['black', '#259c14', '#4c87c6',
                        '#737373', '#cbec24', '#f0441a', '#0d218f']
        plt.figure(figsize=(12, 24))

        for i in range(num_classes):
            plt.subplot(1 * num_classes, 1, i+1)
            plt.plot(densities_list[i, :] * 100, color=class_colors[i])
            # print(densities_list[i, 1:] * 100)
            plt.legend(['Class_' + str(i)])
            plt.xlabel('Classes')
            plt.ylabel('Prediciton')
            plt.ylim(0)

    def batch_dist(self, low_threshold=0.15):
        densities_list = self.get_density()
        batch_dist = densities_list.transpose()
        num_classes = densities_list.shape[1]
        new_dist = []

        for i in range(num_classes):
            class_index = np.where(batch_dist[i, :] > low_threshold)[0]
            new_dist.append(densities_list[class_index, :])

        return new_dist, batch_dist

    # def get_norm_values(self, x_values):
    #     mean = np.mean(x_values)
    #     std = np.std(x_values)
    #     y_values = norm(mean, std).pdf(x_values)

    #     return y_values

    def show_batch_dist(self):
        new_dist, _ = self.batch_dist()
        num_classes = len(new_dist)
        class_colors = ['black', '#259c14', '#4c87c6',
                        '#737373', '#cbec24', '#f0441a', '#0d218f']

        # plt.figure(figsize=(10, 20))
        for j in range(num_classes):
            plt.figure(figsize=(10, 20))
            plt.subplot(num_classes, 1, j+1)
            # j = 4

            sample_x = new_dist[j]
            sample_y = np.zeros(new_dist[j].shape)

            for i in range(num_classes):
                plt.scatter(sample_x[:, i], sample_y[:, i],
                            color=class_colors[i])

                x_values = np.sort(sample_x[:, i])
                mean = np.mean(x_values)
                std = np.std(x_values)

                y_values = norm(mean, std).pdf(np.sort(sample_x[:, i]))

                plt.plot(x_values, y_values, color=class_colors[i])

            plt.legend(['Class_' + str(j)])
            plt.xlabel('Classes')
            plt.ylabel('Prediciton')
            plt.ylim(0)
            plt.show()
