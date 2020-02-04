# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import os
import sys

import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.exposure import equalize_adapthist
from skimage.feature import hog, canny
from skimage.filters import roberts
from skimage.transform import rescale
from tqdm import tqdm

data_dir = "data/"


def crop(inp, crop_percentage=0.1):
    h_cutoff, w_cutoff = (np.array(inp.shape) * crop_percentage).astype(np.int)
    return inp[h_cutoff:-h_cutoff - 1, w_cutoff:-w_cutoff - 1]


def l2_loss(inp, target):
    diff = (inp - target) ** 2
    return np.sum(diff)


def NCC_loss(inp, target):
    inp = inp.flatten()
    target = target.flatten()
    inp /= np.linalg.norm(inp)
    target /= np.linalg.norm(target)
    return -inp @ target


def hog_loss(inp, target):
    inp_h = hog(inp, feature_vector=True)
    tar_h = hog(target, feature_vector=True)
    inp_h /= np.linalg.norm(inp_h)
    tar_h /= np.linalg.norm(tar_h)

    return -inp_h @ tar_h


def roberts_loss(inp, target):
    inp = roberts(inp)
    tar = roberts(target)
    return l2_loss(inp, tar)


def canny_loss(inp, target):
    inp = canny(inp, sigma=2)
    tar = canny(target, sigma=2)
    return l2_loss(inp, tar)


def align(inp, target, center_x=0, center_y=0, search_range=15, crop_percentage=0.1, loss=l2_loss):
    """

    :param loss:
    :param search_range:
    :param inp:
    :param target:
    :return: inp image aligned with target image
    """
    inp_c = crop(inp, crop_percentage)
    target_c = crop(target, crop_percentage)

    curr_min = loss(inp, target)
    curr_shift = (0, 0)

    for i in tqdm(range(center_x - search_range, center_x + search_range + 1)):
        for j in range(center_y - search_range, center_y + search_range + 1):
            shifted = np.roll(inp_c, i, axis=0)
            shifted = np.roll(shifted, j, axis=1)
            diff = loss(shifted, target_c)
            if diff < curr_min:
                curr_min = diff
                curr_shift = (i, j)
    aligned = np.roll(inp, curr_shift[0], axis=0)
    aligned = np.roll(aligned, curr_shift[1], axis=1)
    # print(curr_shift)
    return (aligned,) + curr_shift


def align_pyramid(inp, target, search_range=5, crop_percentage=0.1, loss=l2_loss, base=2.0):
    num_scales = np.ceil(np.log(np.max(inp.shape) / 200) / np.log(base)).astype(np.int)
    print(f"aligning on {num_scales + 1} scales")
    curr_shift = np.array((0, 0), dtype=np.float32)
    for scale in range(num_scales):
        down_samp_inp = rescale(inp, (1 / base) ** (num_scales - scale))
        down_samp_tar = rescale(target, (1 / base) ** (num_scales - scale))
        curr, curr_shift[0], curr_shift[1] = align(down_samp_inp, down_samp_tar, center_x=curr_shift[0].astype(np.int),
                                                   center_y=curr_shift[1].astype(np.int), search_range=search_range,
                                                   crop_percentage=crop_percentage, loss=loss)
        curr_shift *= base
    curr, curr_shift[0], curr_shift[1] = align(inp, target, center_x=curr_shift[0].astype(np.int),
                                               center_y=curr_shift[1].astype(np.int),
                                               search_range=search_range, crop_percentage=crop_percentage, loss=loss)
    return curr, curr_shift[0], curr_shift[1]


def procedure(imname, equalize, used_loss):
    print(f"Processing {imname} with {used_loss.__name__} and equalize = {equalize}")

    base = 1.667
    # read in the image
    im = skio.imread(data_dir + imname)
    # convert to double (might want to do this later on to save memory)
    im = sk.img_as_float(im)
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] // 3).astype(np.int)
    # separate color channels
    b = im[:height]
    g = im[height: 2 * height]
    r = im[2 * height: 3 * height]

    if equalize:
        adj = "adjusted"
        b = equalize_adapthist(b)
        g = equalize_adapthist(g)
        r = equalize_adapthist(r)
    else:
        adj = "original"

    ag, g_shift_y, g_shift_x = align_pyramid(g, b, loss=used_loss, base=base)
    print(f"({int(g_shift_x)}, {int(g_shift_y)})")
    ar, r_shift_y, r_shift_x = align_pyramid(r, b, loss=used_loss, base=base)
    print(f"({int(r_shift_x)}, {int(r_shift_y)})")
    # create a color image
    im_out = np.dstack([ar, ag, b])

    # save the image
    fname = f'out/{used_loss.__name__}-{adj}-({int(g_shift_x)}, {int(g_shift_y)})-' \
            f'({int(r_shift_x)}, {int(r_shift_y)})-{imname.split(".")[0]}.jpg'
    skio.imsave(fname, im_out)

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)
if __name__ == "__main__":
    equalize = sys.argv[1] == "Y"
    for imname in os.listdir(data_dir):
        if imname.endswith("u.tif"):
            for used_loss in (l2_loss, roberts_loss):
                procedure(imname, equalize, used_loss)

# display the image
# skio.imshow(im_out)
# skio.show()
