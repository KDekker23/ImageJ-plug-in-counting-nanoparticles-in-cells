from skimage import util
import cv2
import matplotlib.pyplot as plt
import tifffile
import numpy as np

# Level of noise (noisy or extra_noisy).
noise = 'noisy'
# File path of image to add noise to (PNG or image supported by cv2.imread).
file_image = "something.png"
# Path for saving the noisy image.
save_path = "something.png or .tif"

# Opening the image in grayscale.
# Flags removes the RGB channels, can be disabled by removing the flags argument.
image = cv2.imread(file_image, flags=cv2.IMREAD_GRAYSCALE)

# Applying noise to the image.
if noise == 'noisy':
    noisy_image = util.random_noise(image, mode='gaussian')
elif noise == 'extra_noisy':
    noisy_image = util.random_noise(image, mode='gaussian')
    noisy_image = util.random_noise(noisy_image, mode='speckle', var=4.00)

# Normalize the noisy image and convert to uint8.
to_save = np.array((noisy_image / np.amax(noisy_image) * 255), dtype='uint8')

# Saving the noise image as TIFF or PNG file.
file_type = save_path[save_path.rfind('.') + 1 :]
if file_type == 'tif':
    tifffile.imwrite(save_path, to_save)
elif file_type == 'png':
    cv2.imwrite(save_path, to_save)

# Displaying the normal and noisy image together
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image)
ax[1].imshow(to_save)
plt.show()