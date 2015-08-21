from __future__ import division
import numpy as np

def rescale(x):    
    # note that if python3 division isn't being used, integers will be truncated
    xmin = np.min(x)
    xmax = np.max(x)
    return (x - xmin) / (xmax - xmin)

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
        img_index = self.stimulus[index, 0].astype('int')
        xstart = self.stimulus[index, 1].astype('int')
        ystart = self.stimulus[index, 2].astype('int')
        # Need to check if index is integer or iterable
        try:
            # assume index is a slice
            imgs = [rescale(self.images[img_idx]) for img_idx in img_index]
            return np.array([2*img[y:y+self.ndims, x:x+self.ndims] for x,y,img in zip(xstart,ystart,imgs)])
        except:
            # otherwise index is an integer
            img = rescale(self.images[img_index])
            return img[ystart:ystart+self.ndims, xstart:xstart+self.ndims]

