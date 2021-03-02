#!/usr/bin/env python
# coding: utf-8







# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy
import math
from library import _ni_support
from scipy.ndimage import correlate1d
from skimage.util import img_as_float
from scipy._lib.six import string_types
from skimage import color


# In[3]:


def convolve1d(input, weights, axis=-1, output=None, mode="reflect",
               cval=0.0, origin=0):
    """Calculate a one-dimensional convolution along the given axis.
    The lines of the array along the given axis are convolved with the
    given weights.
    """
    weights = weights[::-1]
    origin = -origin
    if not len(weights) & 1:
        origin -= 1
    return correlate1d(input, weights, axis, output, mode, cval, origin)


# In[4]:


def _gaussian_kernel1d(sigma, radius):
    """
    Computes a 1D Gaussian convolution kernel.
    """
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    
    return phi_x

        


# In[5]:


def gaussian_filter1d(input, sigma, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=4.0):
    """One-dimensional Gaussian filter.
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    
    weights = _gaussian_kernel1d(sigma,lw)
    
    return convolve1d(input, weights, axis, output, mode, cval, 0)


# In[6]:


def gaussian_filter(input, sigma, order=0, output=None,
                    mode="reflect", cval=0.0, truncate=4.0):
    """Multidimensional Gaussian filter.
    """
    input = numpy.asarray(input)
    output = _ni_support._get_output(output, input)
    orders = _ni_support._normalize_sequence(order, input.ndim)
    sigmas = _ni_support._normalize_sequence(sigma, input.ndim)
    modes = _ni_support._normalize_sequence(mode, input.ndim)
    axes = list(range(input.ndim))
    axes = [(axes[ii], sigmas[ii], orders[ii], modes[ii])
            for ii in range(len(axes)) if sigmas[ii] > 1e-15]
    if len(axes) > 0:
        for axis, sigma, order, mode in axes:
            gaussian_filter1d(input, sigma, axis, order, output,
                              mode, cval, truncate)
            input = output
    else:
        output[...] = input[...]
    return output


# In[7]:


def convert_to_float(image, preserve_range):
    """Convert input image to float image with the appropriate range.
    """
    if preserve_range:
        # Convert image to double only if it is not single or double
        # precision float
        if image.dtype.char not in 'df':
            image = image.astype(float)
    else:
        image = img_as_float(image)
    return image


# In[8]:


def _smooth(image, sigma, mode, cval, multichannel=None):
    """Return image with each channel smoothed by the Gaussian filter."""
    smoothed = np.empty(image.shape, dtype=np.double)

    # apply Gaussian filter to all channels independently
    if multichannel:
        sigma = (sigma, ) * (image.ndim - 1) + (0, )
    gaussian_filter(image, sigma, output=smoothed,
                        mode=mode, cval=cval)
    return smoothed


# In[9]:


def _check_factor(factor):
    if factor <= 1:
        raise ValueError('scale factor must be greater than 1')


# In[10]:


import numbers
import numpy as np
from numpy.lib.stride_tricks import as_strided
from warnings import warn

__all__ = ['view_as_blocks', 'view_as_windows']


def view_as_blocks(arr_in, block_shape):
    """Block view of the input n-dimensional array (using re-striding).
    Blocks are non-overlapping views of the input array.
    """
    if not isinstance(block_shape, tuple):
        #raise TypeError('block needs to be a tuple')
        block_shape=np.asarray(block_shape, dtype=int)
        block_shape=tuple(block_shape)
        
    block_shape = np.array(block_shape)
    if (block_shape <= 0).any():
        raise ValueError("'block_shape' elements must be strictly positive")

    if block_shape.size != arr_in.ndim:
        raise ValueError("'block_shape' must have the same length "
                         "as 'arr_in.shape'")

    arr_shape = np.array(arr_in.shape)
    if (arr_shape % block_shape).sum() != 0:
        raise ValueError("'block_shape' is not compatible with 'arr_in'")

    # -- restride the array to build the block view
    new_shape = tuple(arr_shape // block_shape) + tuple(block_shape)
    new_strides = tuple(arr_in.strides * block_shape) + arr_in.strides

    arr_out = as_strided(arr_in, shape=new_shape, strides=new_strides)

    return arr_out


# In[11]:


def block_reduce(image, block_size, func=np.sum, cval=0, func_kwargs=None):
    """Downsample image by applying function `func` to local blocks.
    This function is useful for max and mean pooling, for example.
    """

    if len(block_size) != image.ndim:
        raise ValueError("`block_size` must have the same length "
                         "as `image.shape`.")

    if func_kwargs is None:
        func_kwargs = {}

    pad_width = []
    for i in range(len(block_size)):
        if block_size[i] < 1:
            raise ValueError("Down-sampling factors must be >= 1. Use "
                             "`skimage.transform.resize` to up-sample an "
                             "image.")
        if image.shape[i] % block_size[i] != 0:
            after_width = math.ceil(block_size[i] - (image.shape[i] % block_size[i]))
        else:
            after_width = 0
        pad_width.append((0, after_width))

    image = np.pad(image, pad_width=pad_width, mode='constant',
                   constant_values=cval)
    
    blocked = view_as_blocks(image, block_size)

    return func(blocked, axis=tuple(range(image.ndim, blocked.ndim)),
                **func_kwargs)


# In[12]:


def downscale_local_mean(image, factors, cval=0, clip=True):
    """Down-sample N-dimensional image by local averaging.
    The image is padded with `cval` if it is not perfectly divisible by the
    integer factors.this function calculates the local mean of
    elements in each block of size `factors` in the input image.
    """
    
    return block_reduce(image, factors, np.mean, cval)


# In[13]:


def resize(image, output_shape, order=None, mode='reflect', cval=0, clip=True,
           preserve_range=False, anti_aliasing=None, anti_aliasing_sigma=None):
    """Resize image to match a certain size.Performs down-sampling with integer 
    factor to down-size N-dimensional images. Notet hat anti-aliasing 
    should be enabled when down-sizing images to avoid
    aliasing artifacts.
    """
    output_shape = tuple(output_shape)
    output_ndim = len(output_shape)
    input_shape = image.shape
    if output_ndim > image.ndim:
        # append dimensions to input_shape
        input_shape = input_shape + (1, ) * (output_ndim - image.ndim)
        image = np.reshape(image, input_shape)
    elif output_ndim == image.ndim - 1:
        # multichannel case: append shape of last axis
        output_shape = output_shape + (image.shape[-1], )
    elif output_ndim < image.ndim - 1:
        raise ValueError("len(output_shape) cannot be smaller than the image "
                         "dimensions")

    if anti_aliasing is None:
        anti_aliasing = not image.dtype == bool

    if image.dtype == bool and anti_aliasing:
        warn("Input image dtype is bool. Gaussian convolution is not defined "
             "with bool data type. Please set anti_aliasing to False or "
             "explicitely cast input image to another data type. Starting "
             "from version 0.19 a ValueError will be raised instead of this "
             "warning.", FutureWarning, stacklevel=2)

    factors = np.round((np.asarray(input_shape, dtype=int) /
               np.asarray(output_shape, dtype=int)))

    if anti_aliasing:
        if anti_aliasing_sigma is None:
            anti_aliasing_sigma = np.maximum(0, (factors - 1) / 2)
        else:
            anti_aliasing_sigma =                 np.atleast_1d(anti_aliasing_sigma) * np.ones_like(factors)
            if np.any(anti_aliasing_sigma < 0):
                raise ValueError("Anti-aliasing standard deviation must be "
                                 "greater than or equal to zero")
            elif np.any((anti_aliasing_sigma > 0) & (factors <= 1)):
                warn("Anti-aliasing standard deviation greater than zero but "
                     "not down-sampling along all axes")

        # Translate modes used by np.pad to those used by ndi.gaussian_filter
        np_pad_to_ndimage = {
            'constant': 'constant',
            'edge': 'nearest',
            'symmetric': 'reflect',
            'reflect': 'mirror',
            'wrap': 'wrap'
        }
        try:
            ndi_mode = np_pad_to_ndimage[mode]
        except KeyError:
            raise ValueError("Unknown mode, or cannot translate mode. The "
                             "mode should be one of 'constant', 'edge', "
                             "'symmetric', 'reflect', or 'wrap'. See the "
                             "documentation of numpy.pad for more info.")
        
        image = gaussian_filter(image, anti_aliasing_sigma,
                                    cval=cval, mode=ndi_mode)
        out=downscale_local_mean(image, factors, cval=0, clip=True)

    return out


# In[14]:


def pyramid_reduce(image, downscale=2, sigma=None, order=1,
                   mode='reflect', cval=0, multichannel=False,
                   preserve_range=False):
    """Smooth and then downsample image.
    """
    _check_factor(downscale)

    image = convert_to_float(image, preserve_range)

    out_shape =tuple([math.ceil(d / float(downscale)) for d in image.shape])

    if multichannel:
        out_shape = out_shape[:-1]

    if sigma is None:
        # automatically determine sigma which covers > 99% of distribution
        sigma = 2 * downscale / 6.0
        
    smoothed = _smooth(image, sigma, mode, cval, multichannel)
    out = resize(smoothed, out_shape, order=order, mode=mode, cval=cval,
                 anti_aliasing=True)

    return out


# In[15]:


def pyramid_gaussian(image, max_layer=-1, downscale=2, sigma=None, order=1,
                     mode='reflect', cval=0, multichannel=False,
                     preserve_range=False):
    """Yield images of the Gaussian pyramid formed by the input image.
    Recursively applies the `pyramid_reduce` function to the image, and yields
    the downscaled images.
    Note that the first image of the pyramid will be the original, unscaled
    image. The total number of images is `max_layer + 1`. In case all layers
    are computed, the last image is either a one-pixel image or the image where
    the reduction does not change its shape.
    
    """
    _check_factor(downscale)

    # cast to float for consistent data type in pyramid
    image = convert_to_float(image, preserve_range)

    layer = 0
    current_shape = image.shape

    prev_layer_image = image
    yield image

    # build downsampled images until max_layer is reached or downscale process
    # does not change image size
    while layer != max_layer:
        layer += 1

        layer_image = pyramid_reduce(prev_layer_image, downscale, sigma, order,
                                     mode, cval, multichannel=multichannel)
        
        prev_shape = np.asarray(current_shape)
        prev_layer_image = layer_image
        current_shape = np.asarray(layer_image.shape)

        # no change to previous pyramid layer
        if np.all(current_shape == prev_shape):
            break

        yield layer_image


# In[16]:


def spectrum_gaussian(image, max_layer=-1, downscale=2, sigma=None, order=1,
                     mode='reflect', cval=0, multichannel=False,
                     preserve_range=False):
    """Yield images of the magnitude spectrum of Gaussian pyramid.
    """
    _check_factor(downscale)

    # cast to float for consistent data type in pyramid
    image = convert_to_float(image, preserve_range)

    layer = 0
    current_shape = image.shape

    prev_layer_image = image
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    yield magnitude_spectrum

    # build downsampled images until max_layer is reached or downscale process
    # does not change image size
    while layer != max_layer:
        layer += 1

        layer_image = pyramid_reduce(prev_layer_image, downscale, sigma, order,
                                     mode, cval, multichannel=multichannel)

        prev_shape = np.asarray(current_shape)
        prev_layer_image = layer_image
        current_shape = np.asarray(layer_image.shape)
        f = np.fft.fft2(layer_image)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        # no change to previous pyramid layer
        if np.all(current_shape == prev_shape):
            break

        yield magnitude_spectrum


# In[17]:


def pyramid_laplacian(image, max_layer=-1, downscale=2, sigma=None, order=1,
                      mode='reflect', cval=0, multichannel=False,
                      preserve_range=False):
    """Yield images of the laplacian pyramid formed by the input image.
    Each layer contains the difference between the downsampled and the
    downsampled, smoothed image::
        layer = resize(prev_layer) - smooth(resize(prev_layer))
    Note that the first image of the pyramid will be the difference between the
    original, unscaled image and its smoothed version. The total number of
    images is `max_layer + 1`. In case all layers are computed, the last image
    is either a one-pixel image or the image where the reduction does not
    change its shape.
    
    """
    _check_factor(downscale)

    # cast to float for consistent data type in pyramid
    image = convert_to_float(image, preserve_range)

    if sigma is None:
        # automatically determine sigma which covers > 99% of distribution
        sigma = 2 * downscale / 6.0

    current_shape = image.shape

    smoothed_image = _smooth(image, sigma, mode, cval, multichannel)
    yield image - smoothed_image

    # build downsampled images until max_layer is reached or downscale process
    # does not change image size
    if max_layer == -1:
        max_layer = int(np.ceil(math.log(np.max(current_shape), downscale)))

    for layer in range(max_layer):

        out_shape =tuple([math.ceil(d / float(downscale)) for d in current_shape])

        if multichannel:
            out_shape = out_shape[:-1]

        resized_image = resize(smoothed_image, out_shape, order=order,
                               mode=mode, cval=cval, anti_aliasing=True)
        smoothed_image = _smooth(resized_image, sigma, mode, cval,
                                 multichannel)
        current_shape = np.asarray(resized_image.shape)

        yield resized_image - smoothed_image


# In[18]:


def spectrum_laplacian(image, max_layer=-1, downscale=2, sigma=None, order=1,
                      mode='reflect', cval=0, multichannel=False,
                      preserve_range=False):
    """Yield images of the magnitude spectrum of laplacian pyramid.
    """
    _check_factor(downscale)

    # cast to float for consistent data type in pyramid
    image = convert_to_float(image, preserve_range)

    if sigma is None:
        # automatically determine sigma which covers > 99% of distribution
        sigma = 2 * downscale / 6.0

    current_shape = image.shape

    smoothed_image = _smooth(image, sigma, mode, cval, multichannel)
    
    f = np.fft.fft2(image - smoothed_image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    
    yield magnitude_spectrum

    # build downsampled images until max_layer is reached or downscale process
    # does not change image size
    if max_layer == -1:
        max_layer = int(np.ceil(math.log(np.max(current_shape), downscale)))

    for layer in range(max_layer):

        out_shape = tuple(
            [math.ceil(d / float(downscale)) for d in current_shape])

        if multichannel:
            out_shape = out_shape[:-1]

        resized_image = resize(smoothed_image, out_shape, order=order,
                               mode=mode, cval=cval, anti_aliasing=True)
        smoothed_image = _smooth(resized_image, sigma, mode, cval,
                                 multichannel)
        current_shape = np.asarray(resized_image.shape)
        f = np.fft.fft2(resized_image - smoothed_image)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        yield magnitude_spectrum


# In[19]:


#use grayscale to display clearer laplacian pyramid image
plt.gray()


# In[20]:



image1 = plt.imread('hw2_data/task1and2_hybrid_pyramid/data4.jpg')


fig=plt.figure()
plt.imshow(image1)
image1=color.rgb2gray(image1)

g_pyramid =tuple(pyramid_gaussian(image1, downscale=2, multichannel=False))
g_spec=tuple(spectrum_gaussian(image1, downscale=2, multichannel=False))
l_pyramid=tuple(pyramid_laplacian(image1, downscale=2, multichannel=False))
l_spec=tuple(spectrum_laplacian(image1, downscale=2, multichannel=False))


fig,ax=plt.subplots(4,5,figsize=[128,96])
y=0
for g in g_pyramid[1:6]:

    ax[0,y].imshow(g)
    y+=1
y=0
for l in l_pyramid[1:6]:
    ax[1,y].imshow(l)
    y+=1
y=0
for gs in g_spec[1:6]:
    ax[2,y].imshow(gs)
    y+=1
y=0
for ls in l_spec[1:6]:
    ax[3,y].imshow(ls)
    y+=1

plt.show()    


# In[21]:


from skimage import color
image2 = plt.imread('hw2_data/task1and2_hybrid_pyramid/lara.jpg')


fig=plt.figure()
plt.imshow(image2)
image2=color.rgb2gray(image2)
g_pyramid =tuple(pyramid_gaussian(image2, downscale=2, multichannel=False))
g_spec=tuple(spectrum_gaussian(image2, downscale=2, multichannel=False))
l_pyramid=tuple(pyramid_laplacian(image2, downscale=2, multichannel=False))
l_spec=tuple(spectrum_laplacian(image2, downscale=2, multichannel=False))



fig,ax=plt.subplots(4,5,figsize=[128,96])
y=0
for g in g_pyramid[1:6]:

    
    ax[0,y].imshow(g)
    y+=1
y=0
for l in l_pyramid[1:6]:
    ax[1,y].imshow(l)
    y+=1
y=0
for gs in g_spec[1:6]:
    ax[2,y].imshow(gs)
    y+=1
y=0
for ls in l_spec[1:6]:
    ax[3,y].imshow(ls)
    y+=1

plt.show()    
    


# In[22]:


from skimage import color
image3 = plt.imread('hw2_data/task1and2_hybrid_pyramid/0_Afghan_girl_after.jpg')

fig=plt.figure()
plt.imshow(image3)
image3=color.rgb2gray(image3)

g_pyramid =tuple(pyramid_gaussian(image3, downscale=2, multichannel=False))
g_spec=tuple(spectrum_gaussian(image3, downscale=2, multichannel=False))
l_pyramid=tuple(pyramid_laplacian(image3, downscale=2, multichannel=False))
l_spec=tuple(spectrum_laplacian(image3, downscale=2, multichannel=False))


fig,ax=plt.subplots(4,5,figsize=[128,96])
y=0
for g in g_pyramid[1:6]:

    ax[0,y].imshow(g)
    y+=1
y=0
for l in l_pyramid[1:6]:
    ax[1,y].imshow(l)
    y+=1
y=0
for gs in g_spec[1:6]:
    ax[2,y].imshow(gs)
    y+=1
y=0
for ls in l_spec[1:6]:
    ax[3,y].imshow(ls)
    y+=1

plt.show()    
    


# In[23]:


from skimage import color
image4 = plt.imread('hw2_data/task1and2_hybrid_pyramid/1_motorcycle.bmp')

fig=plt.figure()
plt.imshow(image4)
image4=color.rgb2gray(image4)

g_pyramid =tuple(pyramid_gaussian(image4, downscale=2, multichannel=False))
g_spec=tuple(spectrum_gaussian(image4, downscale=2, multichannel=False))
l_pyramid=tuple(pyramid_laplacian(image4, downscale=2, multichannel=False))
l_spec=tuple(spectrum_laplacian(image4, downscale=2, multichannel=False))


fig,ax=plt.subplots(4,5,figsize=[128,96])
y=0
for g in g_pyramid[1:6]:
    ax[0,y].imshow(g)
    y+=1
y=0
for l in l_pyramid[1:6]:
    ax[1,y].imshow(l)
    y+=1
y=0
for gs in g_spec[1:6]:
    ax[2,y].imshow(gs)
    y+=1
y=0
for ls in l_spec[1:6]:
    ax[3,y].imshow(ls)
    y+=1

plt.show()    

