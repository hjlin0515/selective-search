# coding=utf-8
import skimage.data
import skimage.filters
import skimage.segmentation
import skimage.color
import scipy.ndimage
import numpy as np
import sklearn.preprocessing
import numpy
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

    gaussian = skimage.filters.gaussian(img, sigma=1.0, multichannel=True).astype('float32')

    num_of_orientation = 8
    num_of_bin = 10
    num_of_region = len(set(masks.flatten()))

    r_hist = calculate_texture_hist_per_channel(gaussian[:, :, 0], masks, num_of_region, num_of_orientation, num_of_bin)
    g_hist = calculate_texture_hist_per_channel(gaussian[:, :, 1], masks, num_of_region, num_of_orientation, num_of_bin)
    b_hist = calculate_texture_hist_per_channel(gaussian[:, :, 2], masks, num_of_region, num_of_orientation, num_of_bin)

    texture_hist = np.hstack([r_hist, g_hist, b_hist])
    return sklearn.preprocessing.normalize(texture_hist, norm='l1')


def calculate_color_hist_per_channel(img, masks, num_of_region, num_of_bin):
    bins_of_label = [i for i in range(num_of_region + 1)]
    bins_of_intensity = np.linspace(0.0, 255.0, 26)

    return np.histogramdd(np.vstack([masks.flatten(), img.flatten()]).T,
                          [bins_of_label, bins_of_intensity])[0].reshape(num_of_region, num_of_bin)


def calculate_color_histogram(img, masks):
    num_of_region = len(set(masks.flatten()))
    num_of_bin = 25
    img = skimage.color.rgb2hsv(img)

    r_hist = calculate_color_hist_per_channel(img[:, :, 0], masks, num_of_region, num_of_bin)
    g_hist = calculate_color_hist_per_channel(img[:, :, 1], masks, num_of_region, num_of_bin)
    b_hist = calculate_color_hist_per_channel(img[:, :, 2], masks, num_of_region, num_of_bin)

    color_hist = np.hstack([r_hist, g_hist, b_hist])
    return sklearn.preprocessing.normalize(color_hist, norm='l1')


def extract_regions(img, masks):
    # extract initial regions
    R = {}

    for y, i in enumerate(masks):
        for x, label in enumerate(i):
            if label not in R.keys():
                coor = np.where(masks == label)
                R[label] = {'min_y': np.min(coor[1]), 'max_y': np.max(coor[1]),
                            'min_x': np.min(coor[0]), 'max_x': np.max(coor[0])}


    texture_hist = calculate_texture_histogram(img, masks)
    color_hist = calculate_color_histogram(img, masks)
    sizes = np.bincount(masks.flatten(), minlength=len(set(masks.flatten())))

    for label in R.keys():
        R[label]['color_hist'] = color_hist[label]
        R[label]['texture_hist'] = texture_hist[label]
        R[label]['size'] = sizes[label]

    return R


def calculate_color_similarity(r1, r2):
    return np.sum(np.minimum(r1['color_hist'], r2['color_hist']))


def calculate_texture_similarity(r1, r2):
    return np.sum(np.minimum(r1['texture_hist'], r2['texture_hist']))


def calculate_size_similarity(r1, r2, sizeim):
    return 1 - (r1['size'] + r2['size']) * 1.0 / sizeim


def calculate_fill_similarity(r1, r2, sizeim):
    BB = (max(r1['max_x'], r2['max_x']) - min(r1['min_x'], r2['min_x'])) * \
         (max(r1['max_y'], r2['max_y']) - min(r1['min_y'], r1['min_y']))
    return 1 - (BB - r1['size'] - r2['size']) * 1.0 / sizeim


def calculate_similarities(r1, r2, sizeim, a1=1, a2=1, a3=1, a4=1):
    return a1 * calculate_color_similarity(r1, r2) + a2 * calculate_texture_similarity(r1, r2) + \
           a3 * calculate_size_similarity(r1, r2, sizeim) + a4 * calculate_fill_similarity(r1, r2, sizeim)


def is_neighbour(r1, r2):
    if (r1['max_x'] >= r2['max_x'] >= r2['min_x'] >= r1['min_x'] and r1['max_y'] >= r2['max_y'] >= r2['min_y'] >= r1['min_y'])\
        or (r2['max_x'] >= r1['max_x'] >= r1['min_x'] >= r2['min_x'] and
            r2['max_y'] >= r1['max_y'] >= r1['min_y'] >= r2['min_y']):
        return False

    if (r1['max_y'] > r2['max_y'] > r1['min_y'] and r1['max_x'] > r2['max_x'] > r1['min_x'] ) or\
        (r1['max_y'] > r2['max_y'] > r1['min_y'] and r2['max_x'] > r1['max_x'] > r2['min_x']) or\
        (r2['max_y'] > r1['max_y'] > r2['min_y'] and r1['max_x'] > r2['max_x'] > r1['min_x']) or\
            (r2['max_y'] > r1['max_y'] > r2['min_y'] and r2['max_x'] > r1['max_x'] > r2['min_x']):
        return True
    return False


def merge_region(r1, r2):
    merged = {}
    merged['min_x'] = min(r1['min_x'], r2['min_x'])
    merged['max_x'] = max(r1['max_x'], r2['max_x'])
    merged['min_y'] = min(r1['min_y'], r2['min_y'])
    merged['max_y'] = max(r1['max_y'], r2['max_y'])
    merged['size'] = r1['size'] + r2['size']
    merged['color_hist'] = (r1['size'] * r1['color_hist'] + r2['size'] * r2['color_hist']) / (r1['size'] + r2['size'])
    merged['texture_hist'] = (r1['size'] * r1['texture_hist'] + r2['size'] * r2['texture_hist']) / \
                             (r1['size'] + r2['size'])
    return merged


def selective_search(img, scale=500, sigma=0.9, min_size=10, similarities = ('color', 'texture', 'size', 'fill')):
    """
    :param img:
            scale, sigma, min_size: parameters of felzenszwalb, which decide the initial segments
            similarities: decide which similarities you will use
    :return:[[min_x, max_x, min_y, max_y, size],
            [min_x, max_x, min_y, max_y, size],
            ...
    ]
    """
    a1 = a2 = a3 = a4 = 0
    if 'color' in similarities:
        a1 = 1
    if 'texture' in similarities:
        a2 = 1
    if 'size' in similarities:
        a3 = 1
    if 'fill' in similarities:
        a4 = 1
    # Obtain initial regions
    sizeim = img.shape[0] * img.shape[1]
    masks = skimage.segmentation.felzenszwalb(img, scale, sigma, min_size)

    R = extract_regions(img, masks)
    # calculate initial similarities S
    S = {}
    num_of_region = len(R)
    for i in range(0, num_of_region-1):
        for j in range(i + 1, num_of_region):
            if is_neighbour(R[i], R[j]):
                S[(i, j)] = calculate_similarities(R[i], R[j], sizeim, a1, a2, a3, a4)
    while S != {}:
        # Merge corresponding regions
        i, j = sorted(S.items(), key=lambda x: x[1], reverse=True)[0][0]
        t = len(list(R.keys()))
        R[t] = merge_region(R[i], R[j])

        # Remove similarities regarding ri, rj
        to_be_deleted = []
        for pair in S.keys():
            if (i in pair) or (j in pair):
                to_be_deleted.append(pair)
        for pair in to_be_deleted:
            del S[pair]

        # Calculate Similarity set St between rt and its neighbour
        for index in range(0, t):
            if index != i and index != j:
                if is_neighbour(R[index], R[t]):
                    S[(index, t)] = calculate_similarities(R[t], R[index], sizeim, a1, a2, a3, a4)
    result = []
    for label, region in R.items():
        result.append([region['min_x'], region['max_x'], region['min_y'], region['max_y'], region['size']])

    return result












