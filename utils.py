import os
import numpy as np
from PIL import Image
import base64
from keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt


def preprocess_PIL_img(img_PIL):
    """
    Function: opens a PIL image, resizes and preprocesses it using resnet specification and returns the resulting numpy image tensor
    """ 
    resized_img_PIL = img_PIL.resize((224, 224))
    resized_img_np = np.array(resized_img_PIL)
    preprocessed_img_np = preprocess_input(resized_img_np) # preprocess the image using resnet specifications
    preprocessed_img_np_batch = np.expand_dims(preprocessed_img_np, axis=0)
    return preprocessed_img_np, preprocessed_img_np_batch

def save_mpl_comparison_img(img_PIL, preprocessed_img_np, final_png_path):
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    ax1.imshow(img_PIL, interpolation='nearest')
    ax1.set_title("Your original image:")
    ax2.imshow(preprocessed_img_np, interpolation='nearest')
    ax2.set_title("Preprocessed image as model input:")
    fig.savefig(final_png_path)


def save_file(target_file_path:str, content=None):
    """
    Function: decodes and stores a file uploaded with Plotly Dash.
    """
    data = content.encode("utf8").split(b";base64,")[1]
    with open(target_file_path, "wb") as fp:
        fp.write(base64.decodebytes(data))


def encode_image(image_file):
    """
    Function: encodes a png image
    """
    encoded = base64.b64encode(open(image_file, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded.decode())
