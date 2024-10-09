import os
import cv2
import numpy as np
import math


# --------------------------- Parameter Initialization ---------------------------------------------- #
def initialize_parameters():
    params = {
        'viewtype': "image",
        'output': "none",
        'showquant': "none",
        'auto': "False",
        'poortracking': False,
        'fps': 20,
        'changeinterval': 5,
        'sample_frames': 15,
        'binthresh': 26,
        'xrange': (800, 1200),
        'yrange': (547, 933),
        'bottom_bar': 100,
        'top_bar': 0,
        'left_bar': 0,
        'right_bar': 0,
        'indexlist': [],
        'points_to_image': [],
        'videofile': '../ExampleSplit.avi'
    }

    params['ticker'] = 0
    params['val'] = 0
    params['offset'] = (params['fps'] * params['changeinterval']) * 0.4

    return params


def initialize_collection_frames(params):
    collection_frames = [int((params['fps'] * params['changeinterval'] * i) + params['offset']) for i in range(100)]
    end_collection_frames = [int(frame + params['sample_frames']) for frame in collection_frames]
    return collection_frames, end_collection_frames


def initialize_blob_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 300
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.minInertiaRatio = 0.01
    params.maxInertiaRatio = 0.3

    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        return cv2.SimpleBlobDetector()
    else:
        return cv2.SimpleBlobDetector_create(params)


# -------------------------- Frame and Video Handling ------------------------------------------------ #
def get_frame(cap, frame_num):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    return ret, frame


def initialize_video_capture(params):
    cap = cv2.VideoCapture(params['videofile'])
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
    return cap


def show_frame(frame, y_start, y_end, x_start, x_end):
    cv2.imshow("Frame", frame[y_start:y_end, x_start:x_end])


# -------------------------- Keypoint and Object Tracking -------------------------------------------- #
def detect_keypoints(detector, frame, cleaning_kernel, filling_kernel, rect_pts):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_frame, 26, 255, cv2.THRESH_BINARY)
    clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cleaning_kernel, iterations=1)

    # Apply blocking rectangles
    for rect_pt1, rect_pt2 in rect_pts:
        cv2.rectangle(clean_thresh, rect_pt1, rect_pt2, (0, 0, 0), -1)

    closing = cv2.morphologyEx(clean_thresh, cv2.MORPH_CLOSE, filling_kernel, iterations=2)
    keypoints = detector.detect(closing)
    return keypoints, closing


# -------------------------- Main Processing Loop ---------------------------------------------------- #
def process_video(cap, detector, params, collection_frames, end_collection_frames):
    frame_num = 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Pre-loop setup
    cleaning_kernel = np.ones((2, 2), np.uint8)
    filling_kernel = np.ones((2, 2), np.uint8)
    rect_pts = [((0, 0), (1616, params['top_bar'])), ((0, 0), (params['left_bar'], 1240)),
                ((params['xrange'][1] - params['right_bar'], 0), (params['xrange'][1], 1240)),
                ((0, params['yrange'][1] - params['bottom_bar']), (1616, params['yrange'][1]))]

    # Frame processing loop
    run = True
    while run:
        ret, frame = get_frame(cap, frame_num)
        if not ret:
            break

        keypoints, closing = detect_keypoints(detector, frame, cleaning_kernel, filling_kernel, rect_pts)
        process_keypoints(keypoints)

        # Process frame for tracking and data collection
        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()


def process_keypoints(keypoints):
    for keypoint in keypoints:
        point = keypoint.pt
        # Handle keypoint tracking and height/micro-motion measurements here.
        print("Keypoint detected at:", point)


# -------------------------- Main Program ------------------------------------------------------------ #
def main():
    params = initialize_parameters()
    collection_frames, end_collection_frames = initialize_collection_frames(params)
    cap = initialize_video_capture(params)
    detector = initialize_blob_detector()

    process_video(cap, detector, params, collection_frames, end_collection_frames)


if __name__ == "__main__":
    main()
