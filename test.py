import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

import glob

folder = "test_images/"

images = glob.glob(folder + "*.jpg")

ext_images = [cv2.imread(image) for image in images]


