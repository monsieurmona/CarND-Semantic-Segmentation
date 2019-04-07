import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
import numpy as np
import cv2

def center_crop_image(image, xcrop, ycrop):
    ysize, xsize, chan = image.shape
    xoff = (xsize - xcrop)
    yoff = (ysize - ycrop)

    xto = int(-xoff)
    yto = int(-yoff)

    if (xto == 0):
        xto = xsize

    if (yto == 0):
        yto = ysize

    return image[yoff:yto, xoff:xto]


def center_crop_image2(image, xcrop, ycrop):
    ysize, xsize, chan = image.shape
    xoff = (xsize - xcrop) // 2
    yoff = (ysize - ycrop) // 2

    xto = int(-xoff)
    yto = int(-yoff)

    if (xto == 0):
        xto = xsize

    if (yto == 0):
        yto = ysize

    return image[yoff:yto, xoff:xto]

def random_rotation(image_array_1: ndarray, image_array_2: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-5, 5)
    augmented_image_01 = sk.transform.rotate(image_array_1, random_degree, resize=True)
    augmented_image_02 = sk.transform.rotate(image_array_2, random_degree, resize=True)

    orig_rows, orig_columns, _ = image_array_1.shape

    augmented_image_01 = center_crop_image(augmented_image_01, orig_columns, orig_rows)
    augmented_image_02 = center_crop_image(augmented_image_02, orig_columns, orig_rows)

    augmented_image_01 = sk.transform.resize(augmented_image_01, output_shape=(orig_rows, orig_columns), anti_aliasing=False)
    augmented_image_02 = sk.transform.resize(augmented_image_02, output_shape=(orig_rows, orig_columns), anti_aliasing=False)

    return augmented_image_01, augmented_image_02

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array_1: ndarray, image_array_2: ndarray):
    augumented_image_01 = image_array_1[:, ::-1]
    augumented_image_02 = image_array_2[:, ::-1]

    return augumented_image_01, augumented_image_02


def random_scale(image_array_1: ndarray, image_array_2: ndarray):
    rows, columns, _ = image_array_1.shape

    scale = random.uniform(1, 2)

    resize_rows = int(rows * scale)
    resize_colums = int(columns * scale)

    if (resize_colums == columns and resize_rows == rows):
        return image_array_1, image_array_2

    augmented_image_01 = sk.transform.resize(image_array_1, output_shape=(resize_rows, resize_colums))
    augmented_image_02 = sk.transform.resize(image_array_2, output_shape=(resize_rows, resize_colums))

    augmented_image_01 = center_crop_image2(augmented_image_01, columns, rows)
    augmented_image_02 = center_crop_image2(augmented_image_02, columns, rows)

    new_rows, new_columns, _ = augmented_image_01.shape

    if (new_columns != columns or  new_rows != rows):
        augmented_image_01 = sk.transform.resize(augmented_image_01, output_shape=(rows, columns), anti_aliasing=False)
        augmented_image_02 = sk.transform.resize(augmented_image_02, output_shape=(rows, columns), anti_aliasing=False)

    return augmented_image_01, augmented_image_02

def blur(image, k=7):
  image = cv2.GaussianBlur(image, (k, k), 0)
  return image


def change_illumination(image, sat_limit=(-12, 12), val_limit=(-30, 30)):
  image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
  h, s, v = cv2.split(image)
  sat_shift = np.random.uniform(sat_limit[0], sat_limit[1])
  s = cv2.add(s, sat_shift)
  value_shift = np.random.uniform(val_limit[0], val_limit[1])
  v = cv2.add(v, value_shift)
  image = np.dstack((h, s, v))
  image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
  return image

def distort_image(image, gt_image, collection, gt_collection):
    aug_image, gt_aug_image = random_rotation(image, gt_image)
    aug_image, gt_aug_image = random_scale(aug_image, gt_aug_image)
    aug_image = change_illumination(np.uint8(aug_image * 255))
    gt_aug_image = np.uint8(gt_aug_image * 255)

    collection.append(aug_image)
    gt_collection.append(gt_aug_image)

    return collection, gt_collection

def augment_image(image, gt_image):
    images = []
    gt_images = []

    images.append(image)
    gt_images.append(gt_image)

    for i in range(3):
        distort_image(image, gt_image, images, gt_images)

    flipped_image, gt_flipped_image = horizontal_flip(image, gt_image)

    for i in range(3):
        distort_image(flipped_image, gt_flipped_image, images, gt_images)

    return images, gt_images