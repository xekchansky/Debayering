import numpy as np
from PIL import Image

def create_red_bayer(layer):
    red_bayer = np.zeros(layer.shape)
    for i in range(0, layer.shape[0], 2):
        for j in range(0, layer.shape[1], 2):
            red_bayer[i][j] = layer[i][j]
    return red_bayer

def create_green_bayer(layer):
    green_bayer = np.zeros(layer.shape)
    for i in range(0, layer.shape[0], 2):
        for j in range(1, layer.shape[1], 2):
            green_bayer[i][j] = layer[i][j]
    for i in range(1, layer.shape[0], 2):
        for j in range(0, layer.shape[1], 2):
            green_bayer[i][j] = layer[i][j]
    return green_bayer

def create_blue_bayer(layer):
    blue_bayer = np.zeros(layer.shape)
    for i in range(1, layer.shape[0], 2):
        for j in range(1, layer.shape[1], 2):
            blue_bayer[i][j] = layer[i][j]
    return blue_bayer

def show_bayer_image(bayer_image):
    img = Image.fromarray(bayer_image)
    img.show()
    
def create_bayer_image(image, show=False):
    bayer_image = create_red_bayer(image[:,:,0]) + create_green_bayer(image[:,:,1]) + create_blue_bayer(image[:,:,2])
    if show: show_bayer_image(bayer_image)
    return bayer_image