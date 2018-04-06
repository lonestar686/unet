# preprocess data
import numpy as np
from scipy import ndimage

class DataProcess:
    """ some functions for preprocessing data

    """
    @staticmethod
    def resize(imgs, new_h, new_w, order=1):
        """ resize data
        """
        assert len(imgs.shape) == 4, "has to be in the format of (n, h, w, c)"
        #
        n, h, w, c = imgs.shape[0], imgs.shape[1], imgs.shape[2], imgs.shape[3]

        # scale for new size
        scale_h = new_h/h
        scale_w = new_w/w

        #
        imgs_p = np.ndarray((n, new_h, new_w, c), dtype=np.uint8)

        #
        for i in range(n):
            imgs_p[i] = ndimage.zoom(imgs[i], (scale_h, scale_w, 1), order=order)

        return imgs_p

    @staticmethod
    def scale(imgs_p, flag_simple=True, flag_mask=False):
        """ balance images
        """
        imgs_p = imgs_p.astype('float32')
        print('shape=', imgs_p.shape)

        #
        if flag_simple:
            imgs_p /= 255
        else:
            mean = np.mean(imgs_p)
            std  = np.std(imgs_p)
            imgs_p -= mean
            imgs_p /= std	

        if flag_mask:
            imgs_p[imgs_p > 0.5]  = 1
            imgs_p[imgs_p <= 0.5] = 0

        return imgs_p
