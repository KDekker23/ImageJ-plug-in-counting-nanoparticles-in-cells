import cv2
import numpy as np
import tifffile
from tkinter import filedialog as fd
from os import listdir

# Main name how to save the stack, inverted will be automatically added in the saving process.
# Image stacks will be saved to the chosen directory containing the stack images.
save_name = "image_stack.tif"

# Asking directory from which to access the images that are to be inverted and stacked.
dir = fd.askdirectory()

# Collecting the image paths in a list.
image_files = listdir(dir)

# Collecting the images in a list.
images=[]
for file in image_files:
    image_slice = cv2.imread(dir + "/" + file)
    images.append(image_slice)

# Inverting the images.
inverted_images = []
for image in images:
    inverted_image = ~ image
    inverted_images.append(inverted_image)

# Stacking the images and inverted images.
image_stack = np.stack(images)
inv_image_stack = np.stack(inverted_images)

# Saving the stacks as TIFF files.
tifffile.imwrite(dir + "/" + save_name, image_stack)
tifffile.imwrite(dir + "/inverted_" + save_name, inv_image_stack)