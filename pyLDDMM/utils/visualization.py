import matplotlib.pyplot as plt
import imageio
from PIL import Image

def loadimg(path):
    """
    loads a greyscale image and converts it's datatype
    @param path:
    @return:
    """
    img = imageio.imread(path)
    img_grey = img[:, :, 0]
    return img_grey/ 255.

def saveimg(path, img):
    """
    saves an image
    @param img:
    @param path:
    @return:
    """
    img = Image.fromarray((img * 255).astype('uint8'))
    imageio.imsave(path, img)
    return

def save_animation(path, images):
    """
    creates an animation out of the images
    @param images:
    @return:
    """
    images = [Image.fromarray((images[t] * 255).astype('uint8')) for t in range(len(images))]
    imageio.mimsave(path, images)

def plot_warpgrid(warp, interval=2, show_axis=False):
    """
    plots the given warpgrid
    @param warp: array, H x W x 2, the transformation
    @param interval: int, The interval between grid-lines
    @param show_axis: Bool, should axes be included?
    @return: matplotlib plot. Show with plt.show()
    """
    if show_axis is False:
        plt.axis('off')
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_aspect('equal')

    for row in range(0, warp.shape[0], interval):
        plt.plot(warp[row, :, 1], warp[row, :, 0], 'k')
    for col in range(0, warp.shape[1], interval):
        plt.plot(warp[:, col, 1], warp[:, col, 0], 'k')
    return plt