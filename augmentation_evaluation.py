import scipy.misc
import matplotlib.pyplot as plt
import augment_images as aimg
import numpy as np


image_shape = (160, 576)  # KITTI dataset uses 160x576 images
data_dir = './data'

image = scipy.misc.imresize(scipy.misc.imread('data/data_road/training/image_2/um_000000.png'), image_shape)
gt_image = scipy.misc.imresize(scipy.misc.imread('data/data_road/training/gt_image_2/um_lane_000000.png'), image_shape)

#plt.figure(1)
#plt.imshow(image)
#plt.figure(2)
#plt.imshow(gt_image)


#rot_image, gt_rot_image = aimg.random_rotation(image, gt_image)

#plt.figure("rot")
#plt.imshow(rot_image)
#plt.figure("gt_rot")
#plt.imshow(gt_rot_image)

#scale_image, gt_scale_image = aimg.random_scale(image, gt_image)

#plt.figure("scale")
#plt.imshow(scale_image)
#plt.figure("gt_scale")
#plt.imshow(gt_scale_image)

light_image = aimg.change_illumination(image)

aug_images, gt_aug_images = aimg.augment_image(image, gt_image)

background_color_upper = np.array([255, 0, 0])
background_color_lower = np.array([253, 0, 0])

gt_bg = np.all(np.logical_and(background_color_upper >= gt_aug_images[3],gt_aug_images[3] >= background_color_lower) , axis=2)
gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
gt_augmented_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

print(gt_augmented_image)


plt.show()

