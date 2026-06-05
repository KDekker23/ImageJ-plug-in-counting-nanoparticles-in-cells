import cv2
import tifffile

# Path to PNG image that is to be converted.
png_path = "something.png"
# Save path of how the converted image should be saved.
save_path = "something.tif"

# Opening the images and converting to numpy array
image = cv2.imread(png_path)

# Saving the image, select which image to save and where to save
tifffile.imwrite(save_path, image)