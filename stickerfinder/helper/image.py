"""Some static stuff or helper functions for sticker finder bot."""
import cv2
import numpy as np
import pytesseract

from PIL import Image
from pytesseract import image_to_string
from imutils.object_detection import non_max_suppression

opencv_net = cv2.dnn.readNet("opencv-text-recognition/frozen_east_text_detection.pb")


def preprocess_image(image):
    """Preprocessing the Image for tesseract."""
    # Upscale an image x2
    image = image.resize((4*image.size[0], 4*image.size[1]), resample=Image.LANCZOS)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    return image


def old_get_text_from_image(image):
    """Legacy ocr that doesn't work that good."""
    image = preprocess_image(image)

    # Extract text
    text = image_to_string(image).strip().lower()

    return text


def get_text_from_image(image, config):
    """Execute the whole text recognition process."""
    nparr = np.fromstring(image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # load the input image and grab the image dimensions
    original_image = image.copy()

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (config['width'], config['height']))

    boxes = detect_text_from_boxes(image, config)
    results = extract_text_from_boxes(image, boxes,  original_image, config)

    texts = [result[1] for result in results]

    return ' '.join(texts)


def detect_text_from_boxes(image, config):
    """Detect text in an image.

    This will return a list of rectangles in which opencv detected some text.
    """
    (height, width) = image.shape[:2]
    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    blob = cv2.dnn.blobFromImage(image, 1.0, (width, height),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    opencv_net.setInput(blob)
    (scores, geometry) = opencv_net.forward(layerNames)

    # decode the predictions, then  apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry, config)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    return boxes


def extract_text_from_boxes(image, boxes, original_image, config):
    """Extract the text from the image.

    For each detected box, crop the image to the box and feed
    it into tesseract for text extraction.
    """
    (original_height, original_width) = original_image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    width_ratio = original_width / float(config['width'])
    height_ratio = original_height / float(config['height'])

    # initialize the list of results
    results = []
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * width_ratio)
        startY = int(startY * height_ratio)
        endX = int(endX * width_ratio)
        endY = int(endY * height_ratio)

        # Compute padding delta
        dX = int((endX - startX) * config['padding'])
        dY = int((endY - startY) * config['padding'])

        # Apply padding
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(original_width, endX + (dX * 2))
        endY = min(original_height, endY + (dY * 2))

        # Extract the actual padded ROI
        roi = original_image[startY:endY, startX:endX]

        # Parameters:
        # 1. Set language
        # 2. Use LSTM
        # 3. Use single line ROI
        tesseract_config = ("-l eng --oem 1 --psm 7")
        text = pytesseract.image_to_string(roi, config=tesseract_config)

        # add the bounding box coordinates and OCR'd text to the list
        # of results
        results.append(((startX, startY, endX, endY), text))

    return results


def decode_predictions(scores, geometry, config):
    """Grab the number of rows and columns from the scores volume.

    Then initialize our set of bounding box rectangles and corresponding
    confidence scores
    """
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < config['min_confidence']:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)
