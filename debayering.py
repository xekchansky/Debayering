import numpy as np
from scipy.ndimage.filters import convolve

import GuidedFilter as gf

def interpolate_green(bayer_image):
    green = np.zeros(bayer_image.shape)
    
    #corners
    green[0][0] = (bayer_image[0][1] + bayer_image[1][0]) / 2
    green[0][-1] = bayer_image[0][-1] #(bayer_image[0][-2] + bayer_image[1][-1]) / 2
    green[-1][0] = bayer_image[-1][0] #(bayer_image[-1][1] + bayer_image[-2][0]) / 2
    green[-1][-1] = (bayer_image[-2][-1] + bayer_image[-1][-2]) / 2
    
    #edges
    for i in range(2, bayer_image.shape[0] - 1, 2):
        green[i][0] = (bayer_image[i - 1][0] + bayer_image[i][1] + bayer_image[i + 1][0]) / 3
        green[i - 1][-1] = (bayer_image[i - 1][-1] + bayer_image[i][-2] + bayer_image[i + 1][-1]) / 3

        green[i - 1][0] = bayer_image[i - 1][0]
        green[i][-1] = bayer_image[i][-1]

    for i in range(2, bayer_image.shape[1] - 1, 2):
        green[0][i] = (bayer_image[0][i - 1] + bayer_image[1][i] + bayer_image[0][i + 1]) / 3
        green[-1][i - 1] = (bayer_image[-1][i - 1] + bayer_image[-2][i] + bayer_image[-1][i + 1]) / 3

        green[0][i - 1] = bayer_image[0][i - 1]
        green[-1][i] = bayer_image[-1][i]
        
    #body
    for i in range(1, bayer_image.shape[0] - 1, 2):
        for j in range(1, bayer_image.shape[1] - 1, 2):
            green[i][j] = (bayer_image[i+1][j] + bayer_image[i-1][j] + bayer_image[i][j+1] + bayer_image[i][j-1])/4
            green[i+1][j+1] = (bayer_image[i+2][j+1] + bayer_image[i][j+1] + bayer_image[i+1][j+2] + bayer_image[i+1][j])/4
            green[i+1][j] = bayer_image[i+1][j]
            green[i][j+1] = bayer_image[i][j+1]
            
    green = np.clip(green, 0, 255)
    return green

def get_source_red(bayer_image):
    source_red = np.zeros(bayer_image.shape)
    for i in range(0, bayer_image.shape[0], 2):
        for j in range(0, bayer_image.shape[1], 2):
            source_red[i][j] = bayer_image[i][j]
    return source_red

def get_source_blue(bayer_image):
    source_blue = np.zeros(bayer_image.shape)
    for i in range(0, bayer_image.shape[0], 2):
        for j in range(0, bayer_image.shape[1], 2):
            source_blue[i+1][j+1] = bayer_image[i+1][j+1]
    return source_blue

def interpolate_other(bayer_image, green):
    H = np.array([ [0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25] ])
    
    #red
    source_red = get_source_red(bayer_image)
    tentative_red = gf.GuidedFilter(green, source_red, 2, 1e8)
    red = np.clip(tentative_red + convolve(get_source_red(source_red - tentative_red), H, mode='constant'), 0, 255)
    
    #blue
    source_blue = get_source_blue(bayer_image)
    tentative_blue = gf.GuidedFilter(green, source_blue, 2, 1e8)
    blue = np.clip(tentative_blue + convolve(get_source_blue(source_blue - tentative_blue), H, mode='constant'), 0, 255)
    
    return red, blue

def demosaicking(bayer_image):
    green = interpolate_green(bayer_image)
    red, blue = interpolate_other(bayer_image, green)
    res = np.zeros((bayer_image.shape[0], bayer_image.shape[1], 3))
    res[..., 0] = red.astype(int)
    res[..., 1] = green.astype(int)
    res[..., 2] = blue.astype(int)
    return res.astype(np.uint8)
