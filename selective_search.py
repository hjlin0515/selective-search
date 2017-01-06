# coding=utf-8
import skimage.data
import skimage.filters
import skimage.segmentation
import scipy.ndimage
import numpy as np
import sklearn.preprocessing
from PIL import Image


def calculate_texture_hist_per_channel(img, masks, num_of_region, num_of_orientation, num_of_bin):
    op = np.array([[-1, 0, 1]])
    y = scipy.ndimage.filters.convolve(img, op.T)
    x = scipy.ndimage.filters.convolve(img, op)
    angle = np.arctan2(x, y)

    bins_of_label = [i for i in range(num_of_region + 1)]
    bins_of_angle = np.linspace(-np.pi, np.pi, num_of_orientation + 1)
    bins_of_intensity = np.linspace(0, 1, num_of_bin + 1)

    return np.histogramdd(np.vstack([masks.flatten(), angle.flatten(), img.flatten()]).T,
                          [bins_of_label, bins_of_angle, bins_of_intensity])[0].reshape(num_of_region,
                                                                                        num_of_orientation * num_of_bin)


def calculate_texture_histogram(img, masks):

    gaussian = skimage.filters.gaussian(img, sigma=1.0, multichannel=True)

    num_of_orientation = 8
    num_of_bin = 10
    num_of_region = len(set(masks.flatten()))

    r_hists = calculate_texture_hist_per_channel(gaussian[:, :, 0], masks, num_of_region, num_of_orientation, num_of_bin)
    g_hists = calculate_texture_hist_per_channel(gaussian[:, :, 1], masks, num_of_region, num_of_orientation, num_of_bin)
    b_hists = calculate_texture_hist_per_channel(gaussian[:, :, 2], masks, num_of_region, num_of_orientation, num_of_bin)

    texture_hist = np.hstack([r_hists, g_hists, b_hists])
    return sklearn.preprocessing.normalize(texture_hist, norm='l1')

def calculate_color_histogram()

def extract_regions(img, masks):
    # extract initial regions
    R = {}

    for y, i in enumerate(masks):
        for x, label in enumerate(i):
            if label not in R.keys():
                R[label] = {'min_x': np.inf, 'max_x': -np.inf,
                            'min_y': np.inf, 'max_y': -np.inf}
            if y < R[label]['min_y']:
                R[label]['min_y'] = y
            if x < R[label]['min_x']:
                R[label]['min_x'] = x
            if y > R[label]['max_y']:
                R[label]['max_y'] = y
            if x > R[label]['max_x']:
                R[label]['max_x'] = x

    texture_hist = calculate_texture_histogram(img, masks)
    print(img[masks==1])
    # for i in R.keys():
    #     print(i)






def selective_search(img):
    # Obtain initial regions
    masks = skimage.segmentation.felzenszwalb(image=img, scale=500, sigma=0.9, min_size=70)


    extract_regions(img, masks)



