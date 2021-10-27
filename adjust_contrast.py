# import the necessary packages

from skimage.exposure import is_low_contrast
import numpy as np
import cv2 as cv


def adjust_gamma(image, gamma):
    # build a look up table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0/gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply Gamma correction using the lookup table
    return cv.LUT(image, table)


class Contrast:
    def __init__(self):
        self.gamma = 2.0
        self.image = None
        self.adjusted = None

    def adjust_brightness(self, image):
        self.image = image
        # check to see if the frame is of low contrast
        # and apply adjust_gamma function on each frame
        # and return the frame
        if is_low_contrast(self.image, 0.35):
            self.adjusted = adjust_gamma(self.image, self.gamma)
            return self.adjusted
        # if the frame is of good contrast return the frame
        else:
            return self.image


