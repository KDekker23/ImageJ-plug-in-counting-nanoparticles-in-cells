import numpy as np
import tifffile

# Paths to the images that are to be combined.
cell_path = "path to cell image(stack)"
nano_path = "path to nano image(stack)"
# Path for how to save the new image stack.
save_path = "path to save the new stack"

# Reading the images.
cell_stack = tifffile.imread(cell_path)
nano_stack = tifffile.imread(nano_path)

# Stacking the images to one numpy array.
cell_nano_stack = np.stack((cell_stack, nano_stack))

# Saving the image stack.
tifffile.imwrite(save_path, cell_nano_stack)