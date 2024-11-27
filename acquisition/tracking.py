import os
import cv2
import numpy as np
import math
from tracking_methods import get_frame, set_up_detector, setup_tracking

class TrackingConfig:
    def __init__(self):
        self.VIDEO_FILE = "acquisition/ExampleMicromotion.avi"
        self.VIEW_TYPE = "image"  # Options: "binary", "image", "frame", "cleanthresh"
        self.OUTPUT = "none"      # Options: "tuple", "none"
        self.SHOW_QUANT = "none"  # Options: "height", "micromotion", "both"
        self.POOR_TRACKING = False
        self.FPS = 20
        self.CHANGE_INTERVAL = 5
        self.POINTS_TO_IMAGE = []
        self.SAMPLE_FRAMES = 15
        self.BIN_THRESH = 26
        self.X_RANGE = (100, 900)  
        self.Y_RANGE = (556, 1000)
        self.BOTTOM_BAR = 100
        self.TOP_BAR = 0
        self.LEFT_BAR = 0
        self.RIGHT_BAR = 0
        self.TRACKING_OBJECTS = {}
        self.PIXELCONVERSION = 0.01628


def frame_dimensions(cap, frame_num):
    """
    Calculate frame dimensions and ranges
    :param cap: Video capture object from the OpenCV package
    :param frame_num: Frame number of interest
    :return x_start, x_end,...: Define the rectangular region of interest
    """
    ret, start_frame = get_frame(cap, frame_num)
    start_frame_dim = start_frame.shape
    imageheight = start_frame_dim[0]
    x_start, x_end = config.X_RANGE
    y_start, y_end = (imageheight - config.Y_RANGE[1]), (imageheight - config.Y_RANGE[0])
    return x_start, x_end, y_start, y_end


def gen_initial_frame(cap):
    """
    Generate and display initial frame
    :param cap: Video capture object from the OpenCV package
    :return x_start, x_end,...: Define the rectangular region of interest
    """
    x_start, x_end, y_start, y_end = frame_dimensions(cap, 1)
    ret, start_frame = get_frame(cap, 1)
    cv2.imshow("Frame", start_frame[y_start:y_end, x_start:x_end])
    return x_start, x_end, y_start, y_end


def define_blockers(cap, frame_num):
    """
    Define blocking rectangles for frame processing
    :param cap: Video capture object from the OpenCV package
    :param frame_num: Frame number of interest
    :return: Tuple object containing tuple elements that define the locations of rectangles for omission
    """
    x_start, x_end, y_start, y_end = frame_dimensions(cap, frame_num)
    ylength = y_end - y_start
    xlength = x_end - x_start
    
    top_rect = ((0, 0), (1616, config.TOP_BAR))
    left_rect = ((0, 0), (config.LEFT_BAR, 1240))
    right_rect = ((xlength - config.RIGHT_BAR, 0), (xlength, 1240))
    bottom_rect = ((0, ylength - config.BOTTOM_BAR), (1616, ylength))
    
    return (*top_rect, *left_rect, *right_rect, *bottom_rect)


def post_processing(cap, frame, frame_num):
    """
    Process frame and apply filters
    :param cap: Video capture object from the OpenCV package
    :param frame: Image of the frame returned by cap.read()
    :param frame_num: Frame number of interest
    :return roi_frame: Image of the frame, cropped to the region of interest
    :return closing: Binary image of the frame after erasing small imperfections, cropped to the region of interest
    :return clean_thresh: "Cleaned" image of the frame with small binary imperfections erased
    :return closing_raw: Binary image of the frame post-erasure without the binary blocker
    """
    x_start, x_end, y_start, y_end = frame_dimensions(cap, frame_num)
    blockers = define_blockers(cap, frame_num)
    rectangle_color = (255, 255, 255) if config.VIEW_TYPE == "binary" else (0, 0, 0)
    cleaning_kernel = np.ones((2, 2), np.uint8)
    filling_kernel = np.ones((2, 2), np.uint8)
    roi_frame = frame[y_start:y_end, x_start:x_end]
    gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_frame, config.BIN_THRESH, 255, cv2.THRESH_BINARY)
    clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cleaning_kernel, iterations=1)
    closing_raw = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cleaning_kernel, iterations=1)
    for i in range(0, len(blockers), 2):
        cv2.rectangle(clean_thresh, blockers[i], blockers[i+1], rectangle_color, -1)
    closing = cv2.morphologyEx(clean_thresh, cv2.MORPH_CLOSE, filling_kernel, iterations=2)

    return roi_frame, closing, clean_thresh, closing_raw


def locate_particles(roi_frame, closing, keypoints_prev_frame, frame_num, tracking_objects, track_id, y_end, y_start):
    """
    Locate and track particles in frame
    :param roi_frame: Image of the frame, cropped to the region of interest
    :param closing: Binary image of the frame after erasing imperfections, cropped to the region of interest
    :param keypoints_prev_frame: List object containing tuples of particle locations in the previous frame
    :param frame_num: Frame number of interest
    :param tracking_objects: Dictionary object containing particles' locations
    :param track_id: Index of particle in tracking_objects dictionary
    :param y_end, y_start: Define the y frame of interest
    :return x_position: X position of the particle's centroid, indicated in pixels from the left of the frame of interest
    :return y_position_adj: Y position of the particle's centroid, indicated in pixels from the bottom of the frame of interest
    :return height: Height of detected object in pixels
    :return image_with_keypoints: Image of the frame of interest, red circles drawn at the centroid of detected objects
    :return keypoints_cur_frame: List object containing tuples of particle locations in the current frame
    """
    detector = set_up_detector()
    keypoints = detector.detect(closing)
    keypoints_cur_frame = []
    x_position, y_position_adj, height = "NaN", "NaN", "NaN"
    
    # extract keypoints
    for keypoint in keypoints:
        keypoints_cur_frame.append(keypoint.pt)

    keypoints_copy = keypoints_cur_frame
    
    image_with_keypoints = cv2.drawKeypoints(roi_frame, keypoints, np.array([]), (0, 0, 255))
    
    # find contours
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # track particles
    if frame_num <= 2:
        track_id = _initialize_tracking(keypoints_cur_frame, keypoints_prev_frame, tracking_objects, track_id)
    else:
        _update_tracking(keypoints_cur_frame, tracking_objects)
    
    # process contours and get measurements
    _process_contours(contours, tracking_objects)
    # get position data

    if frame_num >= 2 and len(tracking_objects.keys()) > 0:
        for i in tracking_objects.keys():
            try:
                if tracking_objects[i][1] <= 8:
                    tracking_objects[i].remove
                else:
                    try:
                        if abs(tracking_objects[i][0][0] - keypoints_prev_frame[0][0]) < 2:
                            try:
                                x_position = int(tracking_objects[i][0][0])
                                height = int(tracking_objects[i][1])
                                y_position = int(tracking_objects[i][0][1])
                                y_position_adj = (y_end - y_start) - y_position  # inverts from top-down index to bottom up index
                                for k in range(len(keypoints_copy)):
                                    if x_position == keypoints_copy[k][0]:
                                        keypoints_copy = keypoints_copy[k]
                                        break
                                break
                            except (KeyError, IndexError):
                                pass
                    except IndexError:
                        pass
            except IndexError:
                pass

    return x_position, y_position_adj, height, image_with_keypoints, keypoints_copy


def _initialize_tracking(keypoints_cur_frame, keypoints_prev_frame, tracking_objects, track_id):
    """
    Initialize tracking for new particles
    :param keypoints_cur_frame: List object containing tuples of particle locations in the current frame
    :param keypoints_prev_frame: List object containing tuples of particle locations in the previous frame
    :param tracking_objects: Dictionary object containing particles' locations
    :param track_id: Index of particle in tracking_objects dictionary
    :return track_id : Index of next particle in tracking_objects dictionary
    """
    for pt1 in keypoints_cur_frame:
        for pt2 in keypoints_prev_frame:
            if math.dist(pt1, pt2) < 10:
                tracking_objects[track_id] = [pt1]
                track_id += 1
    return track_id


def _update_tracking(keypoints_cur_frame, tracking_objects):
    """
    Update tracking for existing particles in tracking_objects
    :param keypoints_cur_frame: List object containing tuples of particle locations in the current frame
    :param tracking_objects: Dictionary object containing particles' locations
    """
    tracking_objects_copy = tracking_objects.copy()
    keypoints_cur_frame_copy = keypoints_cur_frame.copy()
    
    for object_id, item2 in tracking_objects_copy.items():
        object_exists = False
        for pt1 in keypoints_cur_frame:
            if math.dist(pt1, item2[0]) < 10:
                tracking_objects[object_id] = [pt1]
                object_exists = True
                if pt1 in keypoints_cur_frame:
                    keypoints_cur_frame.remove(pt1)
                break
        
        if not object_exists:
            tracking_objects.pop(object_id)

    track_id_2 = max(tracking_objects.keys(), default=-1) + 1
    for pt1 in keypoints_cur_frame:
        tracking_objects[track_id_2] = [pt1]
        track_id_2 += 1


def _process_contours(contours, tracking_objects):
    """
    Process contours to get particle dimensions
    :param contours: List object containing contours stored as arrays of points outlining shapes of interest
    :param tracking_objects: Dictionary object containing particles' locations
    """
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        
        for key in tracking_objects.keys():
            if (x <= tracking_objects[key][0][0] <= x + w and 
                y <= tracking_objects[key][0][1] <= y + h):
                tracking_objects[key].append(h)


def analyze_trial(datapoint):
    """
    Analyze trial data and compute averages
    :param datapoint: List object containing tuples (x,y,h) for particles location and height from each of the previous frames
    :return: Tuple reflecting the average of the tuples in datapoint
    """
    if not datapoint:
        return 0, 0, 0
        
    x = [point[0] for point in datapoint]
    y = [point[1] for point in datapoint]
    h = [point[2] for point in datapoint]
    
    return (round(np.mean(x), 2),
            round(np.mean(y), 2),
            round(np.mean(h), 2))


def save_data(yav, hav, frame_num, config, total_frames):
    """
    Puts height and micromotion data (in millimeters, based on PIXELCONVERSION parameter) into Tuple.txt file
    :param yav: Average y-position of the particle over the sample frames, measured from the bottom of the region of interest
    :param hav: Average height of the particle over the sample frames
    :param frame_num: Frame number of interest
    :param config: Class object containing relevant parameters, found at the top of the file
    :param total_frames: Total frames contained in the video object
    :return: Generates or amends the "Tuple.txt" file in the local directory, places list objects formatted as "[yav, hav]" on each line
    """
    try:
        if os.stat('Tuple.txt').st_size != 0 and frame_num <= 70:
            acknowledgement = ""
            while acknowledgement != "continue":
                acknowledgement = input(
                    'Tuple.txt already contains data. Please cancel and clear the file before proceeding. Type "continue" to override')
            print("\ncontinuing...")
        with open('Tuple.txt', 'a') as f:
            yav_mm = yav * config.PIXELCONVERSION
            hav_mm = hav * config.PIXELCONVERSION
            f.write('[' + str(round(yav_mm, 2)) + ', ' + str(round(hav_mm, 2)) + ']\n')
            percentage = (frame_num / total_frames) * 100
            print("Saved: " + str(round(yav_mm, 2)) + ', ' + str(round(hav_mm, 2)) + '; Completion : ' + str(round(percentage, 0)) + '% ' + str(frame_num))
    except FileNotFoundError:
        with open('Tuple.txt', 'w') as f:
            yav_mm = yav * config.PIXELCONVERSION
            hav_mm = hav * config.PIXELCONVERSION
            f.write('[' + str(round(yav_mm, 2)) + ', ' + str(round(hav_mm, 2)) + ']\n')
            percentage = (frame_num / total_frames) * 100
            print("Saved: " + str(round(yav_mm, 2)) + ', ' + str(round(hav_mm, 2)) + '; Completion : ' + str(round((percentage), 0)) + '% ' + str(frame_num))


def auto_run(cap):
    """
    Automatic processing of video frames, outputs datapoints as described below in a Tuple.txt data file
    :param cap: Video capture object from the OpenCV package
    :return: Generates or amends the "Tuple.txt" file in the local directory, places list objects formatted as "[yav, hav]" on each line
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracking_objects, track_id, keypoints_prev_frame = setup_tracking()
    _, _, y_start, y_end = gen_initial_frame(cap)
    
    # calculate collection frames
    collection_frames = [
        int((config.FPS * config.CHANGE_INTERVAL * i) + 
            (config.FPS * config.CHANGE_INTERVAL * 0.4))
        for i in range(100)
    ]
    end_collection_frames = [cf + config.SAMPLE_FRAMES for cf in collection_frames]
    
    _, datapoint = [], []
    collect_data = False
    
    keypoints_prev_frame = []

    # process frames
    for frame_num in range(total_frames):
        ret, frame = get_frame(cap, frame_num)
        tracking_objects, track_id, _ = setup_tracking()
        if not ret:
            break
        roi_frame, closing, _, _ = post_processing(cap, frame, frame_num)
        x, y, h, _, keypoints_cur_frame = locate_particles(roi_frame, closing, keypoints_prev_frame, 
                                 frame_num, tracking_objects, track_id, y_end, y_start)
        keypoints_prev_frame = keypoints_cur_frame
        # collect and analyze data
        if frame_num in collection_frames:
            collect_data = True
        if frame_num in end_collection_frames:
            collect_data = False
            xav, yav, hav = analyze_trial(datapoint)
            save_data(yav, hav, frame_num, config, total_frames)
            datapoint = []
        if collect_data and x != "NaN":
            datapoint.append([x, y, h])


def run_frame(cap, frame_num, keypoints_prev_frame):
    """
    Manually processes and displays each frame. Press a letter or arrow key to progress
    :param cap: Video capture object from the OpenCV package
    :param frame_num: Frame number of interest
    :param keypoints_prev_frame: List object containing tuples of particle locations in the previous frame
    :return frame_num: Frame number of the next frame to analyze
    :return image_with_keypoints: Image of the frame of interest, red circles drawn at the centroid of detected objects
    """
    tracking_objects, track_id, _ = setup_tracking()
    ret, frame = get_frame(cap, frame_num)
    if not ret:
        exit()
    _, _, y_start, y_end = gen_initial_frame(cap)
    roi_frame, closing, _, closing_raw = post_processing(cap, frame, frame_num)
    _, _, _, image_with_keypoints, keypoints_cur_frame = locate_particles(roi_frame, closing, keypoints_prev_frame, 
                                frame_num, tracking_objects, track_id, y_end, y_start)
    
    cv2.imshow("Frame", closing_raw)

    frame_num = frame_num + 1
    return frame_num, keypoints_cur_frame
        

def main():
    """
    Main entry point
    """
    global config
    config = TrackingConfig()
    
    cap = cv2.VideoCapture(config.VIDEO_FILE)
    _, _, _, _ = gen_initial_frame(cap)
    
    frame_num = 0
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        if i == 0:
            keypoints_prev_frame = []
        frame_num, keypoints_prev_frame = run_frame(cap, frame_num, keypoints_prev_frame)
        key = cv2.waitKey()
        if key == 27:  # ESC
            exit()
        if key == 32:  # Space
            auto_run(cap)
        else:
            pass

if __name__ == "__main__":
    main()