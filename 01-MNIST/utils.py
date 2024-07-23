import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


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
    it is displayed using the 'gray' colormap.

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
            plt.imshow(train_set.data[index], cmap='gray')

    plt.show()


def show_confusion_matrix(label_list, pred_list, x_names, y_names,
                          title='Confusion Matrix',
                          color='#e88c13'):
    # confusion matrix
    conf_mat = confusion_matrix(label_list.numpy(), pred_list.numpy())
    accuracy = np.trace(conf_mat) / np.sum(conf_mat).astype('float')
    misclass = 1 - accuracy

    palette = sns.light_palette(color, as_cmap=True)
    ax = plt.subplot()
    sns.heatmap(conf_mat, annot=True, ax=ax, fmt='d', cmap=palette)

    # title
    ax.set_title('\n' + title + '\n',
                 fontweight='bold',  # ['light'|'normal'|'bold'|'heavy']
                 fontstyle='italic',  # ['normal'|'italic'|'oblique']
                 )

    # x y labels
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('Actual', fontweight='bold')

    # labels on axes
    ax.xaxis.set_ticklabels(x_names, ha='center')  # ['left'|'right'|'center']
    ax.yaxis.set_ticklabels(y_names, va='center')  # ['left'|'right'|'center']

    # accuracy and misclassification (1 - accuracy)
    info_text = 'Accuracy={:0.4f} - Misclass={:0.4f}'.format(accuracy, misclass)
    plt.figtext(0.5, 0.05, info_text, ha='center', fontsize=10)

    plt.subplots_adjust(bottom=0.2)

    # show plot
    plt.show()
