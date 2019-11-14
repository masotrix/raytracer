#!/usr/bin/python3
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


img = mpimg.imread('image.png')
imgplot = plt.imshow(img)
plt.show();
