# USAGE
# python scan.py --image images/page.jpg

# import the necessary packages
from dataclasses import dataclass
from typing import Any

from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import ar_markers as ar
import cv2
import imutils
import numpy as np

import pyzbar.pyzbar as pyzbar

screenCnt = []

def click(event, x, y, flag, image):
    global screenCnt
    if event == cv2.EVENT_LBUTTONDOWN:
        coords = [[x, y]]
        screenCnt.append(coords)


def transform_image(image, automatic=False, paper_dims=(1100, 1150), output_image="scannedImage.jpg"):
    """
    :param image: image frame
    :param paper_dims: dimensions of paper (in pixels) to scale scanned image to
    :param output_image: name of file to write new image to
    :return: returns transformation matrix
    """
    global screenCnt
    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    #ratio = image.shape[0] / 500.0
    orig = image.copy()
    #image = imutils.resize(image, height=500)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # show the original image and the edge detected image
    print("STEP 1: Edge Detection")
    cv2.namedWindow('CapturedImage', cv2.WINDOW_NORMAL)
    cv2.imshow("CapturedImage", image)
    cv2.imwrite("image.jpg", image)
    edgedS = cv2.resize(edged, (960, 540))
    #cv2.imshow("Edged", edgedS)


    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    if automatic:
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        # loop over the contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # if our approximated contour has four points, then we
            # can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                break
    else:
        cv2.setMouseCallback('CapturedImage', click, image)
        while(len(screenCnt) < 4):
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                print('Image cropped at coordinates: {}'.format(screenCnt))
                cv2.destroyAllWindows()
                break
        screenCnt = np.asarray(screenCnt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # show the contour (outline) of the piece of paper
    print("STEP 2: Find contours of paper")
    print(screenCnt)
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    imageS = cv2.resize(image, (960, 540))
    cv2.imshow("Outline", imageS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # apply the four point transform to obtain a top-down
    # view of the original image
    M, warped, dims = four_point_transform(orig, screenCnt.reshape(4, 2))
    #find_markers(warped)
    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # T = threshold_local(warped, 11, offset=10, method="gaussian")
    # warped = (warped > T).astype("uint8") * 255

    # show the original and scanned images
    print("STEP 3: Apply perspective transform")

    #cv2.imwrite(output_image, cv2.resize(warped, paper_dims))
    #cv2.imshow("Scanned", cv2.resize(warped, paper_dims))
    cv2.waitKey(0)

    print("Done transform initialization")

    return M, dims


def find_markers(frame, output_image="markers.jpg"):
    """
    :param image_file: filename to read from
    :param output_image: name of file to write new images to
    """

    markers = ar.detect_markers(frame)
    print(frame.shape)
    print(markers)
    for marker in markers:
        marker.highlite_marker(frame)
    #cv2.imshow("Markers", frame)
    #cv2.waitKey(0)
    #cv2.imwrite(output_image, frame)
    return markers


def transform_and_markers(image_file, paper_dims=(825, 1100), scanned_output="scanned.jpg",
                          final_output="scannedMarkers.jpg"):
    """
    :param image_file: original image file to read from
    :param paper_dims: paper dims to scale to (as used in transform image)
    :param scanned_output: output file for image that is scanned but does not have ar markers detected yet
    :param final_output: file name for image with ar markers detect
    """
    transform_image(image_file, paper_dims, scanned_output)
    find_markers(scanned_output, final_output)


def decode(im):
    # Find barcodes and QR codes
    decodedObjects = pyzbar.decode(im)

    # Print results
    for obj in decodedObjects:
        print('Type : ', obj.type)
        print('Data : ', obj.data, '\n')

    return decodedObjects


# Display barcode and QR code location
def highlightCodes(im, decodedObjects):
    # Loop over all decoded objects
    for decodedObject in decodedObjects:
        points = decodedObject.polygon
        print(points)
        # If the points do not form a quad, find convex hull
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points

        # Number of points in the convex hull
        n = len(hull)

        # Draw the convext hull
        for j in range(0, n):
            cv2.line(im, hull[j], hull[(j + 1) % n], (255, 0, 0), 3)

    # Display results
    return im


def find_qr_code(image_file, output_image='qr_code.jpg'):
    im = cv2.imread(image_file)
    decodedObjects = decode(im)
    im2 = highlightCodes(im, decodedObjects)
    cv2.imwrite(output_image, im2)


def transform_and_qr(image_file, paper_dims=(425, 550), scanned_output="scanned.jpg", final_output="scannedQR.jpg"):
    transform_image(image_file, paper_dims, scanned_output)
    find_qr_code(scanned_output, final_output)


@dataclass(frozen=True)
class TransformMetadata:
    transformation_matrix: Any
    im_dims: (float, float)
    desired_dimensions: (float, float)


def transform_point(point: (int, int), transform_metadata: TransformMetadata):
    """
    :param point: point in original plane
    :param M: transformation matrix
    :return: prints point that point is transformed to in new plane
    """
    a = np.array([np.array([point], dtype='float32')])
    cur = cv2.perspectiveTransform(a, transform_metadata.transformation_matrix)
    x = cur.flatten()[0] * transform_metadata.desired_dimensions[0] / transform_metadata.im_dims[0]
    y = cur.flatten()[1] * transform_metadata.desired_dimensions[1] / transform_metadata.im_dims[1]
    return x, y


def get_transform_video(video_path, desired_dimensions=(27.94, 29.21), auto_page_calibrate=True):
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(video_length)
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_length-60)
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    cv2.waitKey(0)
    m, im_dims = transform_image(frame, automatic=auto_page_calibrate)
    return TransformMetadata(m, im_dims, desired_dimensions)

#transform_and_markers("images/arFour.jpg")
'''
dig_markers = find_markers("images/dig_ar_sample.jpg")
transform_and_markers("images/ar_sample.jpg", (816, 1056))
unaltered_markers = find_markers("images/ar_sample.jpg")
my_mat, dims = transform_image("images/ar_sample.jpg", (816, 1056))
print(transform_point(unaltered_markers[0].center, dims, (816, 1056), my_mat))
'''
#transform_metadata = get_transform_video("test_images/test.mp4")
#print(transform_point((591, 263), transform_metadata))
# transform_point([0, 0], my_mat)

def test_angles(orig_img, video_path):
    cap = cv2.VideoCapture(video_path)
    true_markers = find_markers(orig_img)
    gained = []
    #print(true_markers[0].center)
    i = 0
    while(cap.isOpened()):
        i += 1
        ret, frame = cap.read()
        if not ret:
            break
        if i % 30 == 0:
            transform_and_markers(frame, (816, 1056))





#get_transform_video("images/test_vid.mp4")
#cap = cv2.VideoCapture("images/angles.mp4")
#ret, frame = cap.read()
#cv2.imshow('first', frame)
#cv2.waitKey(0)
#print(cap.get(cv2.CAP_PROP_FPS))
#transform_point([0, 0], my_mat)
#test_angles("images/ar_dig.png", "images/angles.mp4")

#get_transform_video("images/englebretson4.MOV", (8.5, 11))