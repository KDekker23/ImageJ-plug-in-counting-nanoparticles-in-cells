# Important imports
import liffile
import numpy as np
import matplotlib.pyplot as plt
import skimage
import cv2
import tifffile
from cellpose import models, plot, core
from os import listdir, mkdir
import PySimpleGUI as sg
import pandas
from scipy.ndimage import gaussian_filter

########## Classes ##########

class Tracker:
    """
    Class responsible for creating a tracker of image slices in an image stack.
    Allows scrolling through an image stack.
    """
    def __init__(self, axis, data, cmap, alpha=1):
        """
        Creates a tracker object and its properties.
        ----------
        Arguments:
            axis -> Axis for plotting images.
            data -> Data of image stack to be plotted.
        """
        self.index = 0
        self.axis = axis
        self.data = data
        # Uses imshow from matplotlib.pyplot to create an image property for the tracker/scroller.
        # Possibility to add cmap here.
        self.image = axis.imshow(self.data[self.index, :, :], cmap=cmap, alpha=alpha)
        # Creating the possibility for the tracker to be updated.
        self.update()
    
    def on_scroll(self, scroll):
        """
        Calculates the next index that is to be represented in the image window.
        -----------
        Arguments:
            scroll -> Whether one scrolled up or down.
        """
        if scroll.button == 'up':
            step = 1
        else:
            step = -1
        max_index = self.data.shape[0] - 1
        # Calculating the new index based on the scroll, current slice position and maximum slice possible.
        self.index = np.clip(self.index + step, 0, max_index)
        # Updating the index.
        self.update()
    
    def update(self):
        """
        Responsible for updating the image presented in the image window,
        based on the calculated new slice. 
        """
        self.image.set_data(self.data[self.index, :, :])
        # +1 to correct for Python indices
        self.axis.set_title(f'slice: {self.index + 1}')
        self.image.axes.figure.canvas.draw()

########## Functions ##########

########## Supporting functions ##########

def check_gpu():
    """
    Checks whether a GPU is available for usage.
    ---------
    Arguments:
        None
    ---------
    Returns:
        gpu -> Bool indicating whether a GPU is available or not.
    """
    gpu = core.use_gpu()
    return gpu

def image_dimensions(image):
    """
    Determines the dimensionality of the data.
    ----------
    Arguments:
        image -> The image array for which the dimensions are to be determined.
    ----------
    Returns:
        data_shape -> Tuple representing the shape of the image.
        dims -> String indicating the number of dimensions present in the image
            and whether channels are present or not.
    """
    data_shape = image.shape
    dimensions = len(data_shape)
    if dimensions == 2:
        dims = '2D'
    elif dimensions == 3:
        if np.amin(data_shape) <= 3:
            dims = '2D_channel'
        else:
            dims = '3D'
    elif dimensions == 4:
        dims = '3D_channel'
    else:
        raise ValueError("Data has incorrect shape to be processed, should be 2D or 3D data.")
    return data_shape, dims

def collect_files(dir):
    """
    Collects the files in a given directory.
    ----------
    Arguments:
        dir -> Path to the directory from which the files should be obtained.
    ----------
    Returns:
        files -> List containing the file names, without the full path, present
            in the directory.
    """
    files = listdir(dir)
    return files

########## Accessing images ##########

def get_images_lif(file_name, channel):
    """
    Function capable of accessing a specific channel of a LIF (Leica Imaging Format) file.
    ----------
    Arguments:
        file_name -> The path towards the LIF file that is to be opened.
        channel -> The channel that should be accessed.
    ----------
    Returns:
        data_stack -> NumPy array containing the image information.
    """
    # Reading the LIF file and accessing the correct image channel.
    lif_image = liffile.imread(file_name)
    # Determining if the file contains 4d or 5d data and based on that
    # access the correct channel of the only or first image.
    if len(lif_image.shape) == 3:
        data_stack = lif_image[channel, :, :]
    elif len(lif_image.shape) == 4:
        data_stack = lif_image[:, channel, :, :]
    elif len(lif_image.shape) == 5:
        data_stack = lif_image[:, 0, channel, :, :]
    return data_stack

def get_images_tif(file_name, channel):
    """
    Function capable of accessing a TIFF file.
    ----------
    Arguments:
        file_name -> Path to the location of the TIFF-file.
    ---------
    Returns:
        image -> NumPy array representing the data present in the image file.
    """
    # Collecting and accessing images
    data_stack = tifffile.imread(file_name)
    im_shape = data_stack.shape
    if len(im_shape) == 2: # Image is 2D without channels
        image = data_stack
    elif len(im_shape) == 3:
        if np.amin(im_shape) <= 3: # Image is channeled 2D image
            image = data_stack[channel]
        else: # Image is 3D image without channels
            image = data_stack
    elif len(im_shape) == 4: # Image is 3D with channels
        # Collecting index of channel axis in dimensions
        chan_ax = np.argmin(im_shape)
        rem_shape = np.delete(im_shape, chan_ax)
        z_ax = np.argwhere(im_shape == np.amin(rem_shape))
        if chan_ax == 0:
            image = data_stack[channel]
        elif chan_ax == 1:
            image = np.einsum('zcxy->czxy', data_stack)[channel]
        elif chan_ax == 3:
            if z_ax == 2:
                image = np.einsum('xyzc->czxy', data_stack)[channel]
            elif z_ax == 0:
                image = np.einsum('zxyc->czxy', data_stack)[channel]
            
        else:
            raise ValueError("Image is of unexpected shape in dimensions: channel axis is not first or second axis.")
        
    return image

def get_images(file_name, cell_chan, nano_chan):
    """
    Accesses the images and stores the cell channel and nano diamond channel
    separately. Is capable of reading TIFF and LIF files, but if the LIF file
    contains more than one channelled image stack, it will only access the 
    first image stack, all others will not be analysed.
    ----------
    Arguments:
        file_name -> The path to the file that is to be analysed. (str)
        cell_chan -> The index of the cell channel. (int)
        nano_chan -> The index of the nano-diamond channel. (int)
    ----------
    Returns:
        cell_image -> NumPy array representing the cell channel of the
            micrograph.
        nano_image -> NumPy array representing the nano-diamond channel
            of the micrograph.
    """
    # Determining the file type of the document chosen.
    file_type = file_name.rfind('.')
    file_type = file_name[file_type + 1 :]  # +1 to remove . from filetype
    # Accessing the images and separating in channel.
    if file_type == 'lif':
        cell_image = get_images_lif(file_name, cell_chan)
        nano_image = get_images_lif(file_name, nano_chan)
    elif file_type == 'tif':
        cell_image = get_images_tif(file_name, cell_chan)
        nano_image = get_images_tif(file_name, nano_chan)
    
    return cell_image, nano_image

########## Segmentation ##########

def cell_segmentation_2d_nc(model, data, batch_size):
    """
    Performs segmentation using Cellpose-SAM on 2D images without channels.
    ---------
    Arguments:
        model -> Loaded or trained model from Cellpose.
        data -> NumPy array representing the cell image.
        batch_size ->  The number of 256x256 patches to run simultaneously on 
            the GPU.
    ---------
    Returns:
        masks -> NumPy array representing the labelled mask of the image.
        flows -> NumPy array containing vector maps representing spatial directions
            of pixels within cells.
        styles -> See Cellpose documentation.
    """
    masks, flows, styles = model.eval(data, batch_size=batch_size)
    return masks, flows, styles

def cell_segmentation_2d_c(dims, model, data, batch_size):
    """
    Performs segmentation using Cellpose-SAM on 2D images with channels.
    ---------
    Arguments:
        dims -> NumPy array displaying image dimensions.
        model -> Loaded or trained model from Cellpose.
        data -> NumPy array representing the cell image.
        batch_size ->  The number of 256x256 patches to run simultaneously on 
            the GPU.
    ---------
    Returns:
        masks -> NumPy array representing the labelled mask of the image.
        flows -> NumPy array containing vector maps representing spatial directions
            of pixels within cells.
        styles -> See Cellpose documentation.
    """
    chan_ax = np.argmin(dims)
    masks, flows, styles = model.eval(data, batch_size=batch_size, channel_axis=chan_ax)
    return masks, flows, styles

def cell_segmentation_3d_nc(dims, model, data, batch_size, stitch_threshold):
    """
    Performs segmentation using Cellpose-SAM on 3D images without channels.
    ---------
    Arguments:
        dims -> NumPy array displaying image dimensions.
        model -> Loaded or trained model from Cellpose.
        data -> NumPy array representing the cell image.
        batch_size ->  The number of 256x256 patches to run simultaneously on 
            the GPU.
        stitch_threshold -> Responsible for linking segmentation objects in 
            different image slices, based on the overlap.
    ---------
    Returns:
        masks -> NumPy array representing the labelled mask of the image.
        flows -> NumPy array containing vector maps representing spatial directions
            of pixels within cells.
        styles -> See Cellpose documentation.
    """
    # Determining the z-axis of the image (smallest axis in the image dimensions).
    z_ax = np.argmin(dims)
    # Performing the segmentation.
    masks, flows, styles = model.eval(data, batch_size=batch_size, do_3D=False, z_axis=z_ax, stitch_threshold=stitch_threshold)
    return masks, flows, styles

def cell_segmentation_3d_c(model, data, batch_size, stitch_threshold):
    """
    Performs segmentation using Cellpose-SAM on 3D images with channels.
    ---------
    Arguments:
        dims -> NumPy array displaying image dimensions.
        model -> Loaded or trained model from Cellpose.
        data -> NumPy array representing the cell image. (CZXY)
        batch_size ->  The number of 256x256 patches to run simultaneously on 
            the GPU.
        stitch_threshold -> Responsible for linking segmentation objects in 
            different image slices, based on the overlap.
    ---------
    Returns:
        masks -> NumPy array representing the labelled mask of the image.
        flows -> NumPy array containing vector maps representing spatial directions
            of pixels within cells.
        styles -> See Cellpose documentation.
    """
    # Performing a 3D segmentation on 3D data.
    masks, flows, styles = model.eval(data, batch_size=batch_size, stitch_threshold=stitch_threshold, z_axis=1, channel_axis=0)
    return masks, flows, styles

def particle_segmentation_2d(data):
    """
    Function capable of segmenting fluorescent nanoparticles. It is not
    based on a specific method of segmentation, but on the reasoning that
    the 10% brightest pixels belong to the nanoparticles. Only works on 2D
    arrays.
    ---------
    Arguments:
        data -> NumPy array representing the image containing the micrograph
            with nanoparticles.
    ---------
    Returns:
        labels -> NumPy array representing a labelled mask, in which each 
            particle is labelled with a different number.
    """
    # Determining minimum and maximum pixel values
    min_val = np.amin(data)
    max_val = np.amax(data)
    # Determine the range of pixelvalues in the image.
    pixel_range = max_val - min_val
    # Set the threshold 10% below the maximum pixel value, so only brightest
    # 10% is in the mask.
    thresh = max_val - np.round(0.1*pixel_range)
    # Apply thresholding and labelling of mask elements.
    mask = data > thresh
    labels = skimage.measure.label(mask, connectivity=2)
    return labels

def particle_segmentation_3d(data):
    """
    Function capable of performing the segmentation of particles in 3D images.
    ----------
    Arguments:
        data -> NumPy array representing the 3D stack of nanodiamond channel.
    ----------
    Returns:
        particle_mask -> NumPy array representing the particle mask, in which
            each particle is labelled with their own number.
    """
    # Determine the minimum and maximum voxel value.
    min_val = np.amin(data)
    max_val = np.amax(data)
    # Determine the range of voxel values.
    voxel_range = max_val - min_val
    # Threshold set at 10% maximum voxel value
    threshold = max_val - np.round(0.25*voxel_range)
    # Performing the segmentation and labelling segmented objects.
    particle_binary = data > threshold
    particle_mask = skimage.measure.label(particle_binary, connectivity=3)
    return particle_mask    

def do_cell_segmentation(data, batch_size, stitch_threshold):
    """
    Function responsible for performing the segmentation of the cell channel. 
    Calls upon other functions, each responsible for a different dimensionality
    of images (2D, 2D with channels, 3D, 3D with channels). 
    ----------
    Arguments:
        data -> NumPy array representing the cell image.
        batch_size -> The number of 256x256 patches to run simultaneously on 
            the GPU.
        stitch_threshold -> Responsible for linking segmentation objects in 
            different image slices, based on the overlap.
    ----------
    Returns:
        cell_mask -> NumPy array representing the labelled mask of cells.
        flows -> NumPy array containing vector maps representing spatial directions
            of pixels within cells.
        styles -> See Cellpose documentation.
    """
    # Checking gpu
    gpu = check_gpu()
    # Collecting dimensionality of the image
    dims = np.array(data.shape)
    # Load Cellpose model for cell segmentation
    model = models.CellposeModel(gpu=gpu, pretrained_model='cpsam')
    # Doing segmentation based on dimensionality.
    if len(dims) == 2:
        cell_mask, flows, styles = cell_segmentation_2d_nc(model, data, batch_size=batch_size)
    elif len(dims) == 3:
        if np.amin(dims) <= 3:
            cell_mask, flows, styles = cell_segmentation_2d_c(dims, model, data, batch_size=batch_size)
        else:
            cell_mask, flows, styles = cell_segmentation_3d_nc(dims, model, data, batch_size=batch_size, stitch_threshold=stitch_threshold)
    elif len(dims) == 4:
        cell_mask, flows, styles = cell_segmentation_3d_c(model, data, batch_size=batch_size, stitch_threshold=stitch_threshold)
    else:
        # NumPy array representing cell image has incorrect shape, thus raise error.
        raise TypeError("Data is of incorrect shape: should be 2D or 3D, and with or without channels (2D to 4D data).")
    return cell_mask, flows, styles

def do_particle_segmentation(data):
    """
    Function responsible for performing the segmentation of particle channel.
    It calls upon other functions for the segmentation, based on the 
    dimensions of the image.
    ----------
    Arguments:
        data -> NumPy array representing the nano-diamond channel of the image.
    ----------
    Returns:
        particle_mask -> NumPy array representing the mask with labelled mask
            elements, each element has their own label value.
    """
    # Collecting dimensionality of the image
    dims = np.array(data.shape)
    # Performing segmentation on the particle images.
    if len(dims) == 2:
        particle_mask = particle_segmentation_2d(data)
    elif len(dims) == 3:
        if np.amin(dims) <= 3:
            particle_mask = particle_segmentation_2d(data)
        else:
            particle_mask = particle_segmentation_3d(data)
    elif len(dims) == 4:
        particle_mask = particle_segmentation_3d(data)
    else:
        raise TypeError("Data is of incorrect shape: should be 2D or 3D, and with or without channels (2D to 4D data).")
    return particle_mask

########## Statistical functions ###########

def cell_vs_particle_2d(cell_mask, part_mask):
    """
    Collects all the image coordinates that contain both fluorescent nano-diamond and cell.
    ----------
    Arguments:
        cell_mask -> The mask made from the cell channel. (NumPy array)
        part_mask -> The mask made from the nano-diamond channel. (NumPy array)
    ----------
    Returns:
        both_pos -> The positions that are present in both the cell mask and the particle
            mask. It is a list containing tuples, representing the image positions.
    """
    # Find all image pixel indices where cells are located.
    cell_indices = np.nonzero(cell_mask)
    cell_row = np.array(cell_indices[0])
    cell_col = np.array(cell_indices[1])
    # Creating a list of tuples of positions.
    cell_pos = []
    for i in range(np.amax(cell_row.shape)): # Find the length of the array as range.
        cell_positions = (int(cell_row[i]), int(cell_col[i]))
        cell_pos.append(cell_positions)

    # Find all image pixel indices where nano-diamonds are located.
    part_indices = np.nonzero(part_mask)
    part_row = np.array(part_indices[0])
    part_col = np.array(part_indices[1])
    # Creating a list of tuples for positions.
    part_pos = []
    for i in range(np.amax(part_row.shape)):
        part_position = (int(part_row[i]), int(part_col[i]))
        part_pos.append(part_position)

    # Collecting the pixel coordinates containing both nano-diamond and cell.
    both_pos = []
    for position in part_pos:
        if position in cell_pos:
            #if position not in both_pos:
            both_pos.append(position)
    return both_pos

def cell_vs_particle_3d(cell_mask, part_mask):
    """
    Function responsible for collecting image indices, where both cell mask
    and particle mask have labelled elements present, indicating overlap in
    cells and particles. Uses cell_vs_particle_2d to collect indices.
    -----------
    Arguments:
        cell_mask -> NumPy array representing the cell mask.
        part_mask -> NumPy array representing the particle mask.
    -----------
    Returns:
        both_positions -> List of 3d tuples representing the image indices
            where both cell and particle is present.
    """
    # Iterating over the mask stacks, to collect identical positions.
    slices = np.amin(cell_mask.shape)
    both_positions = []
    for i in range(slices):
        # Using cell_vs_particle_2d to obtain overlapping positions in one slice.
        cell_mask_slice = cell_mask[i]
        particle_mask_slice = part_mask[i]
        both_positions_2d = cell_vs_particle_2d(cell_mask_slice, particle_mask_slice)
        # Iterating over found positions to make them 3d with the slice number.
        for pos in both_positions_2d:
            slice_index = i
            row_index = pos[0]
            column_index = pos[1]
            position = (slice_index, row_index, column_index)
            both_positions.append(position)
    return both_positions

def locate_particles_2d(cell_mask, part_mask, both_pos, min_size=2, max_size=10):
    """
    Calculates the sizes of potential nano-diamonds inside the cell.
    ----------
    Arguments:
        cell_mask -> NumPy array representing the mask of the cells.
        part_mask -> NumPy array representing the nano-diamond mask.
        both_pos -> pixel coordinates in the image, where both cell and nano-
            diamond is present.
        min_size -> The minimum size in pixels for something to be considered
            as a nano-diamond.
        max_size -> The maximum size in pixels for something to be considered
            as a nanao-diamond.
    ----------
    Returns:
        stats -> Tuple containing multiple numerical values on the position of
            nano-diamonds:
            cells_with_particles -> Nr of cells with nano-diamonds.
            cells_without_particels -> Nr of cells without nano-diamonds.
            average_particles -> The average number of particles in a cell.
            particles_inside -> The number of particles that are inside cells,
                only considered those that are fully inside cells.
            particles_outside -> Nr of particles outside cells, partial overlap
                with a cell is considered to be outside of the cell.
    """
    # Collecting cell and particle labels for matching positions.
    cell_mask_pairs = []
    for position in both_pos:
        cell_label = int(cell_mask[position[0], position[1]])
        part_label = int(part_mask[position[0], position[1]])
        cell_mask_pair = (cell_label, part_label)
        cell_mask_pairs.append(cell_mask_pair)

    # Obtaining unique pairs.
    unique_pairs = list(set(cell_mask_pairs))

    # Calculating the size of the particles:
    particle_sizes = {}
    nr_particles = np.amax(part_mask)
    for i in range(nr_particles):
        # +1 to correct Python counting.
        part_label = i + 1
        size = np.count_nonzero(part_mask == part_label)
        # Only add particles within the correct size range to the list.
        if min_size <= size <= max_size:
            particle_sizes[part_label] = int(size)

    # Counting how many pixels make up the nano-diamond inside the cell.
    diamond_size_cells = []
    for pair in unique_pairs:
        # Check if particle label is belonging to particle, then count pixels in cell belonging to particle.
        if pair[1] in particle_sizes.keys():
            pixels = (pair[1], cell_mask_pairs.count(pair))
            diamond_size_cells.append(pixels)

    # Collecting nano-diamond labels of diamonds fully present in the cell.
    whole_diam_in_cell = []
    for diam_size in diamond_size_cells:
        part_label = diam_size[0]
        full_size_part = particle_sizes[part_label]
        part_in_cell = diam_size[1]
        if part_in_cell == full_size_part:
            whole_diam_in_cell.append(part_label)

    # Collecting cell labels in which nano-diamond is fully present.
    cell_labels = []
    for pair in unique_pairs:
        cell_label = pair[0]
        part_label = pair[1]
        # Check if particle is fully present in the cell.
        if part_label in whole_diam_in_cell:
            cell_labels.append(cell_label)
            
    # Collecting statistics.
    # Determining the number of cells with and without diamonds.
    cells_with_particles = len(cell_labels)
    cells_without_particles = int(np.amax(cell_mask)) - cells_with_particles
    # Calculating the average number of particles in cells (includes cells without particles).
    average_particles = float(len(whole_diam_in_cell) / np.amax(cell_mask))
    # Number of particles inside and outside cells.
    particles_inside = len(whole_diam_in_cell)
    # Using the dictionary to ensure only actual particles are considered.
    particles_outside = len(particle_sizes.keys()) - particles_inside

    stats = (cells_with_particles, cells_without_particles, average_particles, particles_inside, particles_outside)
    return stats

def locate_particles_3d(cell_mask, part_mask, both_pos, min_size=2, max_size=10):
    """
    Function that collects information on the locatoin of particles in the image.
    It functions fully the same as locate_particles_2d, except for the indexing.
    -----------
    Arguments:
        cell_mask -> NumPy array representing the cell mask.
        part_mask -> NumPy array representing the particle mask.
        both_pos -> List containing tuples representing indices that are
            present in both masks.
        min_size -> Minimum size of nano-diamonds in the number of pixels/voxels.
            (Set to 2 automatically)
        max_size -> Maximum size of nano-diamonds in the number of pixels/voxels.
            (Set to 10 automatically)
    ----------
    Returns:
        stats -> Tuple containing numerical results on nano-diamonds.
            cells_with_particles -> The number of cells containing full nanod-diamonds.
            cells_without_particles -> The number of cells that do not contain particles.
            average_particles -> The average number of particles present inside cells.
            particles_inside -> The number of particles inside cells, only consideres the
                particles that are fully inside the cell.
            particles_outside -> The number of particles outside cells, also considered the
                particles that partially overlap with cells.
    """
    # Collecting cell and particle labels for matching positions.
    cell_mask_pairs = []
    for position in both_pos:
        cell_label = int(cell_mask[position[0], position[1], position[2]])
        part_label = int(part_mask[position[0], position[1], position[2]])
        cell_mask_pair = (cell_label, part_label)
        cell_mask_pairs.append(cell_mask_pair)

    # Obtaining unique pairs.
    unique_pairs = list(set(cell_mask_pairs))

    # Calculating the size of the particles:
    particle_sizes = {}
    nr_particles = np.amax(part_mask)
    for i in range(nr_particles):
        # +1 to correct Python counting.
        part_label = i + 1
        size = np.count_nonzero(part_mask == part_label)
        # Only add particles within the correct size range to the list.
        if min_size <= size <= max_size:
            particle_sizes[part_label] = int(size)
    
    # Counting how many pixels make up the nano-diamond inside the cell.
    diamond_size_cells = []
    for pair in unique_pairs:
        # Check if particle label is belonging to particle, then count pixels in cell belonging to particle.
        if pair[1] in particle_sizes.keys():
            # Tuple containing (particle label, nr of pixels for label(in specific label pair))
            pixels = (pair[1], cell_mask_pairs.count(pair))
            diamond_size_cells.append(pixels)

    # Collecting nano-diamond labels of diamonds fully present in the cell.
    whole_diam_in_cell = []
    for diam_size in diamond_size_cells:
        part_label = diam_size[0]
        full_size_part = particle_sizes[part_label]
        part_in_cell = diam_size[1]
        if part_in_cell == full_size_part:
            whole_diam_in_cell.append(part_label)

    # Collecting cell labels in which nano-diamond is fully present.
    cell_labels = []
    for pair in unique_pairs:
        cell_label = pair[0]
        part_label = pair[1]
        # Check if particle is fully present in the cell and if the size is good.
        if part_label in whole_diam_in_cell:
            cell_labels.append(cell_label)
    
    # Collecting statistics.
    # Determining the number of cells with and without diamonds.
    cells_with_particles = len(cell_labels)
    cells_without_particles = int(np.amax(cell_mask) - cells_with_particles)
    # Calculating the average number of particles in cells (includes cells without particles).
    average_particles = float(len(whole_diam_in_cell) / np.amax(cell_mask))
    # Number of particles inside and outside cells.
    particles_inside = len(whole_diam_in_cell)
    # Using the dictionary to ensure that only actual particles are considered.
    particles_outside = len(particle_sizes.keys()) - particles_inside
    
    # Collecting a tuple with all calculated data.
    stats = (cells_with_particles, cells_without_particles, average_particles, particles_inside, particles_outside)
    return stats

########## Full analysis ###########

def execute(dir, file, cell_channel, nano_channel, display_seg, stitch_threshold=0.25, batch_size=8, min_size=2, max_size=10):
    """
    Executes the analysis of an image.
    ------------
     Arguments:
        dir -> The directory containing the images to analyse, which is where
            the results will be saved to after all data is collected. (str)
        file -> The name of the image file which is to be analyzed. (str)
        cell_channel -> Index of the channel containing the cell image. (int)
        nano_channel -> Index of the channel containinng the nano-diamond image.
            (int)
        display_seg -> Whether segmentation results should be displayed or not.
            (bool)
        stitch_threshold -> Responsible for linking segmentation objects in 
            different image slices, based on the overlap. (float)
        batch_size -> The number of 256x256 patches are processed simultaneously.
            (int)
        min_size -> The minimum size of an object to be considered a nano-diamond
            in terms of number of pixels. (int)
        max_size -> The maximum possible size of an image object to be a nano-
            diamond in terms of number of pixels. (int)
    ------------
    Returns:
        cell_mask -> NumPy array representing the mask of the cell channel.
        particle_mask -> NumPy array representing the mask of the particle channel.
        number_results -> Tuple containing the following numbers in order:
            cells_with_particles -> The number of cells containing nano-diamonds.
            cells_without_particles -> The number of cells without nano-diamonds.
            average_particles -> The average number of particles present within cells.
            particles_inside -> The number of particles present inside cells.
            particles_outside -> The number of particles present outside the cells,
                particles that overlap partially with cells are considered as outside
                the cell.
    """
    # Accessing the image and obtaining its dimensions.
    cell_image, particle_image = get_images(dir + '/' + file, cell_channel, nano_channel)
    data_shape, dims = image_dimensions(cell_image)

    # Collecting the cell and nano-diamond masks.
    cell_mask, flows, styles = do_cell_segmentation(cell_image, batch_size=batch_size, stitch_threshold=stitch_threshold)
    particle_mask = do_particle_segmentation(particle_image)
    # Allowing the user to check segmentation results obtained when asked.
    if display_seg == True:
        if dims == '2D' or dims == '2D_channel':
            # Creating a figure and add a title.
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle('Binary masks on top of image.')
            # Plotting the images and the masks in the same figure
            ax[0].imshow(cell_image, cmap='gray')
            ax[0].imshow(cell_mask, cmap='nipy_spectral', alpha=0.25)
            ax[0].set_title('Cell channel')
            ax[1].imshow(particle_image, cmap='gray')
            ax[1].imshow(particle_mask, cmap='nipy_spectral', alpha=0.25)
            ax[1].set_title('Nano-diamond channel')
            plt.show()
        elif dims == '3D' or dims == '3D_channel':
            # Creating the figure on which the images will be displayed.
            fig, ax = plt.subplots(1, 2, figsize=(5, 10))
            # Setting up all the data regarding the cells.
            track_1a = Tracker(ax[0], cell_image, cmap='gray')
            fig.canvas.mpl_connect('scroll_event', track_1a.on_scroll)
            track_1b = Tracker(ax[0], cell_mask, cmap='nipy_spectral', alpha=0.25)
            fig.canvas.mpl_connect('scroll_event', track_1b.on_scroll)
            # Setting up all the data regarding the nano-diamonds.
            track_2a = Tracker(ax[1], particle_image, cmap='gray')
            fig.canvas.mpl_connect('scroll_event', track_2a.on_scroll)
            track_2b = Tracker(ax[1], particle_mask, cmap='nipy_spectral', alpha=0.25)
            fig.canvas.mpl_connect('scroll_event', track_2b.on_scroll)
            plt.show()

    # Collecting the statistics on particle uptake.
    if dims == '2D' or dims == '2D_channel':
        # Comparing cell and particle mask to determine the uptake of nano-diamonds.
        cell_and_particle = cell_vs_particle_2d(cell_mask, particle_mask)
        # Collecting numerical values about particle uptake
        stats = locate_particles_2d(cell_mask, particle_mask, cell_and_particle, min_size=min_size, max_size=max_size)
    elif dims == '3D' or dims == '3D_channel':
        # Comparing cell and particle mask to determine nano-diamond uptake.
        cell_and_particle = cell_vs_particle_3d(cell_mask, particle_mask)
        # Collecting numerical values regarding the partile uptake.
        stats = locate_particles_3d(cell_mask, particle_mask, cell_and_particle, min_size=min_size, max_size=max_size)

    return cell_mask, particle_mask, cell_image, particle_image, stats, dims

########## Saving collected results ##########

def saving_images(dir, file, cell_mask, part_mask, cell_image, dims):
    """
    Function responsible for saving all the obtained results.
    -----------
    Arguments:
        dir -> The directory to which the images should be saved.
        file -> The original file name to which the masks belong.
        cell_mask -> NumPy array representing the mask of the cell channel.
        part_mask -> NumPy array representing the mask of the particle channel.
        cell_image -> NumPy array representing the cell image.
        dims -> The dimensionality of the image, as determined using image
            dimensions function.
    -----------
    Returns:
        Nothing
    """
    # The main path to directory, to which all images will be saved.
    dot_loc = file.rfind('.')
    file_rep = file[: dot_loc]
    dir_path = dir + "/" + file_rep + "/"

    # Creating a folder to store all the images inside the folder.
    mkdir(dir_path)

    # Saving the masks as tiff files.
    c_mask_path = dir_path + 'cell_mask.tif'
    p_mask_path = dir_path + 'particle_mask.tif'
    tifffile.imwrite(c_mask_path, cell_mask) 
    tifffile.imwrite(p_mask_path, part_mask)

    # Saving the masks as bare numpy arrays.
    c_mask_np = dir_path + 'cell_mask.npy'
    p_mask_np = dir_path + 'particle_mask.npy'
    np.save(c_mask_np, cell_mask)
    np.save(p_mask_np, part_mask)

    # Saving the cell mask as a cellpose overlay.
    if dims == '2D' or dims == '2D_channel':
        cell_cellpose = dir_path + 'cellpose_cell_mask.png'
        cellpose_mask = plot.mask_overlay(cell_image, cell_mask)
        cv2.imwrite(cell_cellpose, cellpose_mask)
    elif dims == '3D' or dims == '3D_channel':
        for i, image in enumerate(cell_mask):
            cell_cellpose = dir_path + 'cellpose_cell_mask_' + f'{i}' + '.png'
            cellpose_mask = plot.mask_overlay(cell_image[i], image)
            cv2.imwrite(cell_cellpose, cellpose_mask)

    return None

def collect_numerical_data(files, stats):
    """
    Function responsible for creating a DataFrame containing all the numerical
    information collected with the functions locate_particles 2d and 3d.
    ----------
    Arguments:
        files -> List containing all the files names of the files present in
            the selected directory.
        stats -> Dictionary containing all the numerical data for all the 
            files present in the selected directory.
    ----------
    Returns:
        data_frame -> Pandas DataFrame containing organised numerical data of
            all files analysed.
    """
    # Creating a pandas DataFrame to be able to store the collected numerical values.
    # Collecting all data in listsm by iterating over files and accessing earlier dictionary.
    file_names = []
    cells_w_particles = []
    cells_wn_particles = []
    average_parts = []
    particles_in = []
    particles_out = []
    for file in files:
        file_names.append(file)
        numerical_data = stats[file]
        cells_w_particles.append(numerical_data[0])
        cells_wn_particles.append(numerical_data[1])
        average_parts.append(numerical_data[2])
        particles_in.append(numerical_data[3])
        particles_out.append(numerical_data[4])
    # Creating a dictionary from which the dataframe can be made.    
    data_dictionary = {'file' : file_names,
                       'cells with nano-diamonds' : cells_w_particles,
                       'cells without nano-diamonds' : cells_wn_particles,
                       'average nano-diamond uptake' : average_parts,
                       'nano-diamonds in cells' : particles_in,
                       'nano-diamonds outside cells' : particles_out}
    # Creating a dataframe, which can be saved.
    data_frame = pandas.DataFrame(data_dictionary)
    
    return data_frame

def save_numerical_data(dir, data_frame):
    """
    Function responsible for saving a pandas DataFrame as a CSV file.
    ----------
    Arguments:
        dir -> The directory to which the CSV file should be saved.
        data_frame -> The pandas DataFrame containing the collected
            numerical information about the images.
    ----------
    Returns:
        None
    """
    # Setting a path to where the data should be saved.
    save_path = dir + '/' + 'numerical_info.csv'
    # Saving the pandas DataFrame as a csv file.
    data_frame.to_csv(save_path, index=False)
    return None

########## GUI setup ###########

def run_gui():
    """
    Function responsible for running the program. It creates the GUI
    and runs all the functions, meaning it executes the program. The
    GUI is build using PySimpleGUI. This function does not take any
    arguments and it only returns 'finished'.
    """
    # Defining initial window layout
    layout_a1 = [[sg.Text("Select directory containing images for analysis:", key='directory')],
                 [sg.Input(key='dir'), sg.FolderBrowse()],
                 [sg.Button('Confirm'), sg.Button('Exit')]]

    # Creating the window for layout a1.
    window = sg.Window("Welcome to particle counting", layout_a1)

    # Collecting initial values.
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'exit':
            break
        elif event == 'Confirm':
            dir = values['dir']
            window.close()

    # Collecting the files in the directory.
    files = collect_files(dir)
    # Creating a string of the file names, which aid in the display of selected files.
    string_files = str(files[0])
    for i, file in enumerate(files):
        if i == 0:
            string_files = string_files
        else:
            string_files = string_files + '\n' + str(file)

    # Changing the layout to contain the selected files.
    layout_a2 = [[sg.Text(f"The selected directory is: {dir}")],
                 [sg.Text("Which contains the files: ")],
                 [sg.Multiline(key='files', size=(30, len(files)), default_text=string_files)],
                 [sg.Text("Continue with these files?"), sg.Button('Yes'), sg.Button('No')]]

    # Setting the window for layout a2.
    window = sg.Window("Welcome to particle counting", layout_a2)

    # Collecting interaction user, whether to start the program.
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        elif event == 'Yes':
            start_intiation = True
            window.close()
        elif event == 'No':
            start_intiation = False
            window.close()
            run_gui()
            
    # Updating the layout of the window again, to ask for the last inputs.
    # Layout of the main input tab.
    tab_layout_1a = [[sg.Text("What is the cell channel?"), sg.Checkbox('Channel 0', key='cell_0'), sg.Checkbox('Channel 1', key='cell_1')],
                    [sg.Text("What is the nano-diamond channel?"), sg.Checkbox('Channel 0', key='nano_0'), sg.Checkbox('Channel 1', key='nano_1')],
                    [sg.Text("Is the image / are the images 2d or 3d?"), sg.ButtonMenu('Dimension', ['Dimension', ['2D', '3D']], key='dimension')],
                    [sg.Text("It is recommended to check the quality of segmentation before processing an image batch.")],
                    [sg.Text("For this it is best to process one image (2d or 3d) and check the segmentation for the image.")],
                    [sg.Text("Do you wish to check segmentation?"), sg.Button('Yes'), sg.Button('No')]]
    # Layout of the second input tab.
    tab_layout_1b = [[sg.Text("Stitch threshold (between 0 and 1)"), sg.Input('0.25', key='stitch_threshold')],
                    [sg.Text("Batch size (integer)"), sg.Input('8', key='batch_size')],
                    [sg.Text("Minimum size nano-diamond 2d (integer number of pixels)"), sg.Input('2', key='min_size_2d')],
                    [sg.Text("Maximum size nano-diamond 2d (integer number of pixels)"), sg.Input('10', key='max_size_2d')],
                    [sg.Text("Minimum size nano-diamond 3d (integer number of voxels)"), sg.Input('6', key='min_size_3d')],
                    [sg.Text("Maximum size nano-diamond 3d (integer number of voxels)"), sg.Input('30', key='max_size_3d')]]
    # Layout of the window.
    layout_a3 = [[sg.TabGroup([[sg.Tab("Main input", tab_layout_1a), sg.Tab("Additional input", tab_layout_1b)]])]]

    # Setting the window using layout a3.
    if start_intiation == True:
        window = sg.Window("Welcome to particle counting", layout_a3)
    else:
        run_gui()

    # Displaying the updated window and collecting the last input.
    while True:
        # Based on interaction decide how the program continues.
        event, values = window.read()
        if event == 'Yes':
            display_segmentation = True
            window.close()
        elif event == 'No':
            display_segmentation = False
            window.close()
        elif event == sg.WIN_CLOSED:
            break
        # Collecting channel information.
        if values['cell_0'] == True:
            cell_channel = 0
        else:
            cell_channel = 1
        if values['nano_0'] == True:
            nano_channel = 0
        else:
            nano_channel = 1

        # Collecting remaining settings.
        # Stitch threshold.
        if values['stitch_threshold'] != '0.25':
            stitch_threshold = float(values['stitch_threshold'])
        else:
            stitch_threshold = 0.25
        # Batch size.
        if values['batch_size'] != '8':
            batch_size = int(values['batch_size'])
        else:
            batch_size = 8
        # Minimum nano-diamond size (2d).
        if values['min_size_2d'] != '2':
            min_size_2d = int(values['min_size_2d'])
        else:
            min_size_2d = 2
        # Maximum nano-diamond size (2d).
        if values['max_size_2d'] != '10':
            max_size_2d = int(values['max_size_2d'])
        else:
            max_size_2d = 10
        # Minimum nano-diamond size (3d).
        if values['min_size_3d'] != 6:
            min_size_3d = int(values['min_size_3d'])
        else:
            min_size_3d = 10
        # Maximum nano-diamond size (3d).
        if values['max_size_3d'] != 30:
            max_size_3d = int(values['max_size_3d'])
        else:
            max_size_3d = 30

        # Collecting dimensionality of data.
        if values['dimension'] == '2D':
            min_size = min_size_2d
            max_size = max_size_2d
            dimension = '2D'
        elif values['dimension'] == '3D':
            min_size = min_size_3d
            max_size = max_size_3d 
            dimension = '3D'   

    # First tab displaying the main settings and information collected.
    tab_layout_2a = [[sg.Text(f"The selected directory is: {dir}")],
                    [sg.Text(f'The cells are in channel: {cell_channel}, and the nano-diamonds in channel: {nano_channel}.')],
                    [sg.Text(f'Segmentation will be displayed: {display_segmentation}.')],
                    [sg.Text(f'Data has the shape {dimension}.')],
                    [sg.Text("Is all the information correct?")],
                    [sg.Text("Selecting 'Yes' will start the program.")],
                    [sg.Button('Yes'), sg.Button('No')]]
    # Second tab displaying the extra settings.
    tab_layout_2b = [[sg.Text(f'Stitch threshold: {stitch_threshold}')],
                     [sg.Text(f'Batch size: {batch_size}')],
                     [sg.Text(f'Minimum size nano-diamonds 2d: {min_size_2d} pixels')],
                     [sg.Text(f'Maximum size nano-diamonds 2d: {max_size_2d} pixels')],
                     [sg.Text(f'Minimum size nano-diamonds 3d: {min_size_3d} voxels')],
                     [sg.Text(f'Maximum size nano-diamonds 3d: {max_size_3d} voxels')]]
    # Updating the window to display the initally collected data.
    layout_a4 = [[sg.TabGroup([[sg.Tab("Main input", tab_layout_2a), sg.Tab("Additional input", tab_layout_2b)]])]]
    # Creating the last start window.
    window = sg.Window("Checking all input", layout_a4)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        elif event == 'Yes':
            start_running = True
            window.close()
        elif event == 'No':
            start_running = False
            window.close()
            run_gui()
            
    # Running the code when the user indicated start running.
    # Collecting the statistics about each image in a dictionary.
    statistics_dict = {}
    if start_running == True:
        for file in files:
            #file_name = dir + '/' + file
            #cell_image, particle_image = get_images(file_name, cell_channel, nano_channel)
            # Getting the masks and the numerical information.
            c_mask, p_mask, c_image, p_image, stats, dims = execute(dir=dir, file=file, cell_channel=cell_channel, nano_channel=nano_channel, display_seg=display_segmentation,
                                                  stitch_threshold=stitch_threshold, batch_size=batch_size, min_size=min_size, max_size=max_size)
            # Saving the numerical results in a dictionary
            statistics_dict[f"{file}"] = stats
            # Saving the masks in multiple forms to a directory with the image name.
            saving_images(dir, file, c_mask, p_mask, c_image, dims)
    # Collecting all numerical data in a DataFrame.        
    data_frame = collect_numerical_data(files, statistics_dict)
    # Saving the DataFrame as a CSV file.
    save_numerical_data(dir, data_frame)

    return "Finished"

########## End of functions for program #########


# Statement that enables running of the program.
print(run_gui())


########## Functions capable of smoothing ##########

"""
It was forgotten to built in smoothing functions to the program, but they were 
written, which is why these functions are here. To apply the image smoothing, 
one simply can alter the code using the following steps:
1. go to the funtion get_images (lines 186-215)
2. go to the section that collects the images (lines 209, 210, 212, 213)
3. change: image = get_images_lif(...) to: image = image_smoothing(get_images_lif(...))
4. same method applies to the collecting of tif files.
5. save the program file
Now you can run the program again, but this time image smoothing is applied to
the images that are to be analysed.

To restore the program to the original state, you simply change it back to:
image = get_images_lif(...) or images = get_images_tif(...)

It was not added to the program, eventhough it is a simple adjustment, because
otherwise all the test images would have to be rerun for the report that was
made for this project, and there was not enough time to do so.
"""

def image_smoothing(data):
    """
    Function capable of smoothing the images, to reduce the noise present.
    ----------
    Arguments:
        data -> NumPy array representing the image that is to be smoothed.
    ----------
    Returns:
        smoother_image -> NumPy array representing the smoothed image, which
            can be used in further processing steps to improve visibility.
    """
    # Determine data shape
    dims = data.shape
    if len(dims) == 2:
        dimension = '2D'
        channel_ax = None
    elif len(dims) == 3:
        if np.amin(dims) <= 3:
            dimension = '2D'
            channel_ax = 0
        else:
            dimension = '3D'
            channel_ax = None
    elif len(dims) == 4:
        dimension = '3D'
        channel_ax = 0

    # Estimate the noise standard deviation, using data shape and smooth the image
    # The sigma is divided to reduce over-smoothing effects.
    if channel_ax == None:
        est_sigma = skimage.restoration.estimate_sigma(data, channel_axis=None)
        smoother_image = skimage.restoration.denoise_wavelet(data, sigma=est_sigma/2, channel_axis=None, method='VisuShrink')
    elif channel_ax == 0:
        est_sigma = skimage.restoration.estimate_sigma(data, channel_axis=channel_ax)
        smoother_image = skimage.restoration.denoise_wavelet(data, sigma=est_sigma/2, channel_axis=channel_ax, method='VisuShrink')
    return smoother_image

def image_preparation(data, normalization=True, sigma=2):
    """
    Function responsible for applying preprocessing to the image.
    ----------
    Arguments:
        data -> Image or imagestack to which the preprocessing should be applied.
            (numpy array)
        normalization -> Whether normalization should be applied to the image. 
            It is standard set to True, which means normalization is applied, 
            when it shouldn't be applied, this is to be False.
        sigma -> The number of standard deviations for the size of the Gaussian
            filter kernel.
    ----------
    Returns:
        preprocessed_image -> The image or image stack after preprocessing has 
            been applied. (numpy array)
    """
    # Applying normalization, if wanted, and Gaussian smoothing.
    if normalization == True:
        normalized_image = cv2.normalize(data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        preprocessed_image = gaussian_filter(normalized_image, sigma)
    else:
        preprocessed_image = gaussian_filter(data, sigma)
    return preprocessed_image