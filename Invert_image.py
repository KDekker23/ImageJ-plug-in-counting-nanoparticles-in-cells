import cv2
import tifffile

# Path to the image
png_path = "something.png"
# How to save the result (complete path and what the file should be called.)
save_path = ".tif or .png"

# Accessing the images
# Flags removes the three colour channels. Remove flags argument if channels should be kept.
np_image = cv2.imread(png_path, flags=cv2.IMREAD_GRAYSCALE)

# Inverting the images
inverted_image = ~ np_image

# Saving the inverted image based on extension of save_path.
file_type = save_path[save_path.rfind('.') + 1 :]
if file_type == 'tif':
    tifffile.imwrite(save_path, inverted_image)
elif file_type == 'png':
    cv2.imwrite(save_path, inverted_image)
