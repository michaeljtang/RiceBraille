# import the necessary packages
import numpy as np
import argparse
import cv2 as cv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image")
args = vars(ap.parse_args())

# load the image
image = cv.imread("images/blurNight.jpg")
cv.namedWindow("Display Window", cv.WINDOW_AUTOSIZE)
# Take each frame
# Convert BGR to HSV
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
# define range of blue color in HSV
delta = 2.5
lower_green = np.array([46 - delta, 100, 100])
upper_green = np.array([46 + delta, 255, 255])

lower_blue = np.array([100 - delta, 100, 100])
upper_blue = np.array([100 + delta, 255, 255])

lower_pink = np.array([166 - delta, 90, 200])
upper_pink = np.array([166 + delta, 200, 255])

lower_purple = np.array([130 - delta, 0, 0])
upper_purple = np.array([130 + delta, 255, 255])

lower_yellow = np.array([28 - delta, 0, 0])
upper_yellow = np.array([28 + delta, 255, 255])

lower_purple_two = np.array([146 - delta, 0, 0])
upper_purple_two = np.array([146 + delta, 255, 255])


# Threshold the HSV image to get only blue colors
# Threshold the HSV image to get only green colors
mask_green = cv.inRange(hsv, lower_green, upper_green)
# Threshold for blue
mask_blue = cv.inRange(hsv, lower_blue, upper_blue)
mask_pink = cv.inRange(hsv, lower_pink, upper_pink)
mask_purple = cv.inRange(hsv, lower_purple, upper_purple)
mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)

def maskMaker(mask_values, delta_h, delta_s):
    mask = False
    for val in mask_values:
        cur_top = np.array([(val[0]*0.5)+delta_h, (val[1]*255.0/100)+delta_s, (val[2]*255.0/100)+delta_s])
        cur_bot = np.array([(val[0]*0.5)-delta_h, (val[1]*255.0/100)-delta_s, (val[2]*255.0/100)-delta_s])
        cur_mask = cv.inRange(hsv, cur_bot, cur_top)
        mask = mask | cur_mask
    return mask

'''
green = (94, 61, 89)
blue = (196, 90, 90)
purple = (259, 36, 100)
pink = (332, 37, 100)
'''
red = (360, 72, 47)
orange = (18, 71, 64)
purple_two = (318, 71, 43)
yellow = (50, 64, 64)
my_vals = [red, orange, purple_two, yellow]
# Bitwise-AND mask and original image
res = cv.bitwise_and(image, image, mask=(maskMaker(my_vals, 10, 50)))
print(image.shape)
cv.imwrite("original.jpg", image)
cv.imwrite("mask.jpg", mask_blue)
cv.imwrite("res.jpg", res)
'''
cv.imshow('Display Window', image)
cv.waitKey(0)
cv.imshow('mask', mask)
cv.waitKey(0)
cv.imshow('res', res)
cv.waitKey(0)
'''


