import numpy as np
import skimage as sk
import skimage.io as skio
from scipy.signal import convolve2d
import cv2

data_dir = "data/"
imname = "cameraman.png"

def imshow(img):
    skio.imshow(img)
    skio.show()

Dx = np.array([[1, -1]])
Dy = Dx.copy().T

im = skio.imread(data_dir + imname, as_gray=True)
im = sk.img_as_float(im)

print(im.shape)

# Part 1.1
im_dx = convolve2d(im, Dx, mode="same")
im_dy = convolve2d(im, Dy, mode="same")
im_gradmag = (im_dx**2 + im_dy**2)**0.5

skio.imshow(im_gradmag >=0.25)
skio.show()

# Part 1.2
gkern = cv2.getGaussianKernel(5, 2)
gfilter = gkern @ gkern.T

im_dx_smooth = convolve2d(im_dx, gfilter, mode="same")
im_dy_smooth = convolve2d(im_dy, gfilter, mode="same")
im_gradmag_smooth = (im_dx_smooth**2 + im_dy_smooth**2)**0.5
skio.imshow(im_gradmag_smooth >=0.1)
skio.show()

DoGx = convolve2d(gfilter, Dx)
DoGy = convolve2d(gfilter, Dy)

im_DoGx = convolve2d(im, DoGx, mode="same")
im_DoGy = convolve2d(im, DoGy, mode="same")
im_DoG_mag = (im_DoGx**2 + im_DoGy**2)**0.5
skio.imshow(im_DoG_mag >= 0.1)
skio.show()

# Part 1.3
facade = skio.imread(data_dir + "facade.jpg", as_gray=True)
skio.imshow(facade)
skio.show()

f_Dogx = convolve2d(facade, DoGx, mode="same")
f_Dogy = convolve2d(facade, DoGy, mode="same")
f_DoG_mag = (f_Dogx**2 + f_Dogy**2)**0.5
imshow(f_DoG_mag >= 0.05)