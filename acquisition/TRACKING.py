import os
import cv2
import numpy as np
import math

# --------------------------- Constants Definition ---------------------------------------------- #
VIDEO_FILE = "../ExampleSplit.avi"
VIEW_TYPE = "image"  # Options: "binary", "image", "frame", "cleanthresh"
OUTPUT = "none"      # Options: "tuple", "none"
SHOW_QUANT = "none"  # Options: "height", "micromotion", "both"
POOR_TRACKING = False # Ignoring index of the particle if varying
FPS = 20
CHANGE_INTERVAL = 5
POINTS_TO_IMAGE = [] # Frames to capture
SAMPLE_FRAMES = 15
BIN_THRESH = 26
X_RANGE = (800, 1200)
Y_RANGE = (547, 933)
BOTTOM_BAR = 100
TOP_BAR = 0
LEFT_BAR = 0
RIGHT_BAR = 0
INDEX_LIST = [] # Indices of desired tracking particles

# ------------------------------- Helper Functions ---------------------------------------------- #

def initialize_video(cap):
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cleaning_kernel = np.ones((2, 2), np.uint8)
    filling_kernel = np.ones((2, 2), np.uint8)
    return total_frames, cleaning_kernel, filling_kernel

def setup_detector():
    params = cv2.SimpleBlobDetector.Params()
    # Filter by color
    params.filterByColor = True
    params.blobColor = 255
    # Filter by area (pixels)
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 300
    # Filter by circularity
    params.filterByCircularity = False
    # Filter by convexity
    params.filterByConvexity = False
    # Filter by inertia ratio (To detect elongated shapes)
    params.filterByInertia = False
    params.minInertiaRatio = 0.01
    params.maxInertiaRatio = 0.3
    return cv2.SimpleBlobDetector_create(params)

def get_frame(cap, got_frame_num):
    cap.set(cv2.CAP_PROP_POS_FRAMES, got_frame_num)
    ret, got_frame = cap.read()
    return ret, got_frame

def frame_dimensions(cap, frame_num):
    ret, start_frame = get_frame(cap, frame_num)
    start_frame_dim = start_frame.shape
    imageheight = start_frame_dim[0]
    x_start, x_end = X_RANGE
    y_start, y_end = (imageheight - Y_RANGE[1]), (imageheight - Y_RANGE[0])
    return x_start, x_end, y_start, y_end

def gen_initial_frame(cap):
    total_frames, cleaning_kernel, filling_kernel = initialize_video(cap)
    x_start, x_end, y_start, y_end = frame_dimensions(cap, 1)
    ret, start_frame = get_frame(cap, 1)
    cv2.imshow("Frame", start_frame[y_start:y_end, x_start:x_end])
    return x_start, x_end, y_start, y_end

def define_blockers(frame_num):
    x_start, x_end, y_start, y_end = frame_dimensions(frame_num)
    ylength = y_end - y_start
    xlength = x_end - x_start
    top_rect_pt1, top_rect_pt2 = (0, 0), (1616, TOP_BAR)
    left_rect_pt1, left_rect_pt2 = (0, 0), (LEFT_BAR, 1240)
    right_rect_pt1, right_rect_pt2 = (xlength - RIGHT_BAR, 0), (xlength, 1240)
    bottom_rect_pt1, bottom_rect_pt2 = (0, ylength - BOTTOM_BAR), (1616, ylength)
    return (top_rect_pt1, top_rect_pt2, left_rect_pt1, left_rect_pt2, right_rect_pt1, right_rect_pt2, bottom_rect_pt1, bottom_rect_pt2)

def post_processing(cap, frame, frame_num):
    x_start, x_end, y_start, y_end = frame_dimensions(cap, frame_num)
    ylength = y_end - y_start
    xlength = x_end - x_start
    top_rect_pt1, top_rect_pt2 = (0, 0), (1616, TOP_BAR)
    left_rect_pt1, left_rect_pt2 = (0, 0), (LEFT_BAR, 1240)
    right_rect_pt1, right_rect_pt2 = (xlength - RIGHT_BAR, 0), (xlength, 1240)
    bottom_rect_pt1, bottom_rect_pt2 = (0, ylength - BOTTOM_BAR), (1616, ylength)
    if VIEW_TYPE == "binary":
        rectangle_color = (255, 255, 255)
    else:
        rectangle_color = (0, 0, 0)
    cleaning_kernel = np.ones((2, 2), np.uint8)
    filling_kernel = np.ones((2, 2), np.uint8)
    roi_frame = frame[y_start:y_end, x_start:x_end]
    gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_frame, BIN_THRESH, 255, cv2.THRESH_BINARY)
    clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cleaning_kernel, iterations=1)
    cv2.rectangle(clean_thresh, top_rect_pt1, top_rect_pt2, rectangle_color, -1)  # Top Erasure
    cv2.rectangle(clean_thresh, left_rect_pt1, left_rect_pt2, rectangle_color, -1)  # Left Erasure
    cv2.rectangle(clean_thresh, right_rect_pt1, right_rect_pt2, rectangle_color, -1)  # Right Erasure
    cv2.rectangle(clean_thresh, bottom_rect_pt1, bottom_rect_pt2, rectangle_color, -1)  # Bottom erasure
    closing = cv2.morphologyEx(clean_thresh, cv2.MORPH_CLOSE, filling_kernel, iterations=2)
    return roi_frame, closing, clean_thresh

def setup_tracker():
    tracking_objects = {}
    track_id = 0
    keypoints_prev_frame = []
    return tracking_objects, track_id, keypoints_prev_frame

def locate_particles(roi_frame, closing, keypoints_prev_frame, frame_num, tracking_objects, track_id, y_end, y_start):
    detector = setup_detector()
    keypoints = detector.detect(closing)
    keypoints_cur_frame = []
    x_position, y_position, height = "NaN", "NaN", "NaN"
    for keypoint in keypoints:
        keypoints_cur_frame.append(keypoint.pt)
    image_with_keypoints = cv2.drawKeypoints(roi_frame, keypoints, np.array([]), (0, 0, 255))
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if frame_num <= 2:
        for pt1 in keypoints_cur_frame:
            for pt2 in keypoints_prev_frame:
                distance = math.dist(pt1, pt2)
                if distance < 10:
                    tracking_objects[track_id] = [pt1]
                    track_id += 1
    else:
        tracking_objects_copy = tracking_objects.copy()
        keypoints_cur_frame_copy = keypoints_cur_frame.copy()
        for object_id, item2 in tracking_objects_copy.items():
            object_exists = False
            for pt1 in keypoints_cur_frame:
                distance = math.dist(pt1, item2[0])
                if distance < 10:
                    tracking_objects[object_id] = [pt1]
                    object_exists = True
                    if pt1 in keypoints_cur_frame:
                        keypoints_cur_frame.remove(pt1)
                    continue
            if not object_exists:
                tracking_objects.pop(object_id)
    for pt1 in keypoints_cur_frame:
        tracking_objects[track_id] = [pt1]
        track_id += 1
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        centroid_x, centroid_y = int(x + w / 2), int(y + h / 2)
        adj_centroid_y = (y_end - y_start) - centroid_y
        for key in tracking_objects.keys():
            if x <= tracking_objects[key][0][0] <= x + w and y <= tracking_objects[key][0][1] <= y + h:
                tracking_objects[key].append(h)
    if frame_num >= 2:
        try:
            if len(tracking_objects.keys()) > 0:
                try:
                    x_position, y_position, height = int(tracking_objects[0][0][0]), int(tracking_objects[0][0][1]), int(tracking_objects[0][1])
                except KeyError:
                    pass
        except KeyError or IndexError:
            pass
    return x_position, y_position, height

def analyze_trial(datapoint):
    x, y, h = [], [], []
    for i in range(len(datapoint)):
        x.append(datapoint[i][0])
        y.append(datapoint[i][1])
        h.append(datapoint[i][2])
    xav, yav, hav = round(np.mean(x), 2), round(np.mean(y), 2), round(np.mean(h), 2)
    return xav, yav, hav

def auto_run(cap):
    total_frames, cleaning_kernel, filling_kernel = initialize_video(cap)
    tracking_objects, track_id, keypoints_prev_frame = setup_tracker()
    x_start, x_end, y_start, y_end = gen_initial_frame(cap)
    collection_frames = [int((FPS * CHANGE_INTERVAL * i) + (FPS * CHANGE_INTERVAL * 0.4)) for i in range(100)]
    end_collection_frames = [cf + SAMPLE_FRAMES for cf in collection_frames]
    trial, datapoint = [], []
    collect_data = False
    for frame_num in range(total_frames):
        ret, frame = get_frame(cap, frame_num)
        roi_frame, closing, clean_thresh = post_processing(cap, frame, frame_num)
        x, y, h = locate_particles(roi_frame, closing, keypoints_prev_frame, frame_num, tracking_objects, track_id, y_end, y_start)
        if frame_num in collection_frames:
            collect_data = True
        if frame_num in end_collection_frames:
            collect_data = False
            xav, yav, hav = analyze_trial(datapoint)
            trial.append([xav, yav, hav])
            datapoint = []
        if collect_data and x != "NaN":
            datapoint.append([x, y, h])


# --------------------------- Main Functionality ---------------------------------------------- #
def main():
    cap = cv2.VideoCapture(VIDEO_FILE)
    x_start, x_end, y_start, y_end = gen_initial_frame(cap)
    key = cv2.waitKey()
    if key == 27:
        exit()
    if key == 32:
        auto_run(cap)


if __name__ == "__main__":
    main()
