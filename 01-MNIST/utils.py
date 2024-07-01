import cv2
import numpy as np
from matplotlib import pyplot as plt


# Define our imshow function
def img_show(title="", image=None, size=6):
    """
    Display an image with a specified title and size.
    This function uses matplotlib to display the image. The image is expected to
    be in BGR format (common in OpenCV), and will be converted to RGB format for
    displaying. The aspect ratio of the image will be maintained.

    :param title: str, optional: The title of the image to be displayed (default
        is an empty string).
    :param image: numpy.ndarray, optional: The image to be displayed. It should
        be a valid image in the form of a NumPy array (default is None).
    :param size: int or float, optional: The size of the displayed image in
        inches. The height of the image will be set to this value, and the width
        will be adjusted according to the aspect ratio of the image (default is
        6).
    """
    if image is None:
        return

    # check if the image is grayscale or rgb
    if len(image.shape) == 3:
        h, w, d = image.shape  # height, width, depth
    else:
        h, w = image.shape  # shape[0] -> height, width

    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


def show_collage(train_set, size=(15, 8)):
    """
    Display a collage of the first 50 images from a given Torch dataset.

    This function arranges the first 50 images from the `train_set` in a 5x10
    grid. It displays each image in the grid without axes. If the image is in
    RGB format, it is displayed in color. If the image is in grayscale format,
    it is displayed using the 'gray_r' colormap.

    :param size: tuple, optional: The size of the displayed image in inches.
    :param train_set: object: PyTorch dataset.
    """
    plt.figure(figsize=(size[0], size[1]))

    num_of_images = 50

    for index in range(1, num_of_images + 1):
        plt.subplot(5, 10, index)
        plt.axis('off')
        if len(train_set.data[index].shape) == 3:
            plt.imshow(train_set.data[index])
        else:
            plt.imshow(train_set.data[index], cmap='gray_r')

    plt.show()
