from __future__ import division
import numpy as np
import collections

def rolling_window(array, window):
    """
    Make an ndarray with a rolling window of the last dimension
    Parameters
    ----------
    array : array_like
        Array to add rolling window to
    window : int
        Size of rolling window
    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.
    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])
    Calculate rolling mean of last dimension:
    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])
    """
    assert window >= 1, "`window` must be at least 1."
    assert window < array.shape[-1], "`window` is too long."

    # # with strides
    shape = array.shape[:-1] + (array.shape[-1] - window, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

class NaturalScenesStimulus(object):
    '''
    Take the compressed natural scenes data and return the appropriate frame(s).
    Callable like a numpy array, even though it generates the stimulus behind
    the scenes.
    
    INITIALIZE NaturalScenesStimuli with the following arguments: 
    images:     Array of images with shape (num_images, height, width)
    stimulus:   Array of shape (num_frames, 3) where the three dimensions refer
                to (image_index, xstart, ystart). Image_index should be in
                python format (0,N-1) and NOT matlab format (1,N).

    Then this class allows you to call stimulus frames as if it's a numpy array.

    E.g. if dataset is an HDF5 file 'f' with datasets 'images' and 'expt1',
    where 'expt1/stim' has the stimulus parameters (img_idx, xstart, ystart) for
    each frame, then you could access every other frame from 5 to 10 just as if
    it's a normal numpy array:
    
    natural_stimuli = NaturalScenesStimulus(f['images'], f['expt1/stim'])
    stimuli_subset = natural_stimuli[5:10:2]
    '''

    def __init__(self, images, stimulus):
        '''
        Images should be (num_images, height, width).
        Stimulus should be (num_frames, 3) where the three
        dimensions refer to image index, xstart, ystart.
        '''
        self.images = images
        self.stimulus = stimulus
        self.ndims = 500
        self.shape = (stimulus.shape[0], self.ndims, self.ndims)

    def __getitem__(self, index):
        '''
        Returns np array of shape (index.shape[0], self.ndims, self.ndims)
        '''
        img_index = self.stimulus[index, 0]
        xstart = self.stimulus[index, 1]
        ystart = self.stimulus[index, 2]
        # Need to check if index is integer or iterable
        if isinstance(img_index, np.ndarray):
            assert img_index.dtype is np.dtype('int16'), 'img_index is %s' %(img_index.dtype)
            assert xstart.dtype is np.dtype('int16'), 'xstart is %s' %(xstart.dtype)
            assert ystart.dtype is np.dtype('int16'), 'ystart is %s' %(ystart.dtype)
            # assume index is a slice
            imgs = [self.images[img_idx] for img_idx in img_index]
            return np.array([2*img[y:y+self.ndims, x:x+self.ndims] for x,y,img in zip(xstart,ystart,imgs)])
        else:
            # otherwise index is an integer
            img = self.images[img_index]
            return img[ystart:ystart+self.ndims, xstart:xstart+self.ndims]

class NaturalDatasetStimulus(object):
    '''
    Take the compressed natural scenes data and return the appropriate examples,
    after Toeplitz matrix reordering.

    Callable like a numpy array, even though it generates the stimulus behind
    the scenes.
    
    INITIALIZE NaturalScenesStimuli with the following arguments: 
    images:     Array of images with shape (num_images, height, width)
    stimulus:   Array of shape (num_frames, 3) where the three dimensions refer
                to (image_index, xstart, ystart). Image_index should be in
                python format (0,N-1) and NOT matlab format (1,N).
    duration:   Number of frames per example (e.g. 40 frames at 10 ms would mean
                a 400ms filter length).

    Then this class allows you to call stimulus frames as if it's a numpy array.
    '''

    def __init__(self, frames, duration):
        '''
        Frames is a NaturalScenesStimulus() object.

        Duration is length of filter in frames.
        '''
        self.duration = duration
        self.frames = frames
        self.shape = (frames.shape[0]-duration+1, self.duration, frames.shape[1], frames.shape[2])

    def __getitem__(self, index):
        '''
        Returns np array of shape (index.shape[0], self.ndims, self.ndims)
        '''
        indices = np.arange(self.shape[0])[index]

        # case where you want a single example
        if ~isinstance(indices, np.ndarray):
            indices = np.array([indices])

        X = np.zeros((indices.shape[0],) + self.shape[1:])
        for i in indices:
            ex_indices = np.arange(i,i+self.duration)
            X[i] = self.frames[ex_indices]
            
            return X

class NaturalDatasetSpikes(object):
    '''
    Return the spikes aligned to NaturalDatasetStimulus object.
    '''
    def __init__(self, spikes, duration):
        '''
        Spikes is ndarray of shape (ncells, nbins)

        Duration is length of filter in frames. Should be same
        as NaturalDatasetStimulus.duration.
        '''
        self.ncells = spikes.shape[0]
        self.duration = duration
        self.spikes = spikes[:,duration-1:]

    def __getitem__(self, index):
        '''
        Returns np array of shape (index.shape[0],).
        '''
        return self.spikes[index]



