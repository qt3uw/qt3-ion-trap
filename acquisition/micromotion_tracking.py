import os
import cv2
import numpy as np
import math
from tracking_methods import get_frame, set_up_detector, setup_tracking


class MicromotionTrackingConfig:
    def __init__(self):
        self.video_file = "acquisition/ExampleMicromotion.avi"
        self.view_type = "image"        # "image" to block out white binary noise, "binary" to block out black binary noise
        self.start_frame = 1600            # Defines starting frame. ONLY FOR DEBUGGING
        self.fps = 20                   # fps of the camera
        self.start_voltage = 40         # Initial voltage value 
        self.voltage_increment = 5      # Voltage step between datapoints
        self.change_interval = 5        # Time between data points in the real-time trial (seconds)
        self.sample_frames = 15         # Number of frames averaged over per data point
        self.bin_thresh = 26            # Binary threshold for object detection
        self.x_range = (200, 900)       # x-axis frame of interest limits
        self.y_range = (554, 1000)      # y-axis frame of interest limits
        self.bottom_bar = 100           # Erasure rectangle, measured in pixels from the bottom edge
        self.top_bar = 0                # Erasure rectangle, measured in pixels from the top edge
        self.left_bar = 0               # Erasure rectangle, measured in pixels from the left edge
        self.right_bar = 0              # Erasure rectangle, measured in pixels from the right edge
        self.pixel_to_mm = 0.01628      # Pixel-to-millimeter conversion, gathered from calibration image. "None" will output raw pixel data


def get_default_config():
    return MicromotionTrackingConfig()


def frame_dimensions(cap, frame_num, config = get_default_config()):
    """
    Calculate frame dimensions and ranges
    :param cap: Video capture object from the OpenCV package
    :param frame_num: Frame number of interest
    :return x_start, x_end,...: Define the rectangular region of interest
    """
    ret, initial_frame = get_frame(cap, frame_num)
    initial_frame_dim = initial_frame.shape
    imageheight = initial_frame_dim[0]
    x_start, x_end = config.x_range
    y_start, y_end = (imageheight - config.y_range[1]), (imageheight - config.y_range[0])
    return x_start, x_end, y_start, y_end


def gen_initial_frame(cap, config = get_default_config()):
    """
    Generate and display initial frame
    :param cap: Video capture object from the OpenCV package
    :return x_start, x_end,...: Define the rectangular region of interest
    """
    frame_num = config.start_frame
    x_start, x_end, y_start, y_end = frame_dimensions(cap, frame_num)
    ret, initial_frame = get_frame(cap, config.start_frame)
    cv2.imshow("Frame", initial_frame[y_start:y_end, x_start:x_end])
    return x_start, x_end, y_start, y_end


def define_blockers(cap, frame_num, config = get_default_config()):
    """
    Define blocking rectangles for frame processing
    :param cap: Video capture object from the OpenCV package
    :param frame_num: Frame number of interest
    :return: Tuple object containing tuple elements that define the locations of rectangles for omission
    """
    x_start, x_end, y_start, y_end = frame_dimensions(cap, frame_num)
    ylength = y_end - y_start
    xlength = x_end - x_start
    
    top_rect = ((0, 0), (1616, config.top_bar))
    left_rect = ((0, 0), (config.left_bar, 1240))
    right_rect = ((xlength - config.right_bar, 0), (xlength, 1240))
    bottom_rect = ((0, ylength - config.bottom_bar), (1616, ylength))
    
    return (*top_rect, *left_rect, *right_rect, *bottom_rect)


def post_processing(cap, frame, frame_num, config = get_default_config()):
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
    rectangle_color = (255, 255, 255) if config.view_type == "binary" else (0, 0, 0)
    cleaning_kernel = np.ones((2, 2), np.uint8)
    filling_kernel = np.ones((2, 2), np.uint8)
    roi_frame = frame[y_start:y_end, x_start:x_end]
    gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_frame, config.bin_thresh, 255, cv2.THRESH_BINARY)
    clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cleaning_kernel, iterations=1)
    closing_raw = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cleaning_kernel, iterations=1)
    for i in range(0, len(blockers), 2):
        cv2.rectangle(clean_thresh, blockers[i], blockers[i+1], rectangle_color, -1)
    closing = cv2.morphologyEx(clean_thresh, cv2.MORPH_CLOSE, filling_kernel, iterations=2)

    return roi_frame, closing, clean_thresh, closing_raw


def locate_particles(roi_frame, closing, keypoints_prev_frame, frame_num, tracking_objects, track_id, y_end, y_start, last_known = None, config = get_default_config()):
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
    for keypoint in keypoints:
        keypoints_cur_frame.append(keypoint.pt)
    keypoints_copy = keypoints_cur_frame
    image_with_keypoints = cv2.drawKeypoints(roi_frame, keypoints, np.array([]), (0, 0, 255))
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if frame_num <= config.start_frame + 2:
        track_id = _initialize_tracking(keypoints_cur_frame, keypoints_prev_frame, tracking_objects, track_id)
    else:
        _update_tracking(keypoints_cur_frame, tracking_objects)
    _process_contours(contours, tracking_objects)
    if frame_num >= (config.start_frame + 2) and len(tracking_objects.keys()) > 0:
        tracking_objects_copy = tracking_objects.copy()
        for i in tracking_objects_copy.keys():
            try:
                if tracking_objects[i][1] <= 10:
                    tracking_objects.pop(i)
                elif last_known != 0 and last_known is not None:
                    if abs(last_known[0][0] - tracking_objects[i][0][0]) >= 5 or \
                            abs(last_known[0][1] - tracking_objects[i][0][1]) >= 5:
                        tracking_objects.pop(i)
                    else:
                        x_position = int(tracking_objects[i][0][0])
                        height = int(tracking_objects[i][1])
                        y_position = int(tracking_objects[i][0][1])
                        y_position_adj = (y_end - y_start) - y_position
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
    :return track_id: Index of next particle in tracking_objects dictionary
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
    :return: Updated tracking_objects dictionary for the current frame
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


def save_data(yav, hav, frame_num, total_frames, datapoint_num, config = get_default_config()):
    """
    Puts height and micromotion data (in millimeters, based on pixel_to_mm parameter) into text file
    :param yav: Average y-position of the particle over the sample frames, measured from the bottom of the region of interest
    :param hav: Average height of the particle over the sample frames
    :param frame_num: Frame number of interest
    :param total_frames: Total frames contained in the video object
    :param datapoint_num: Datapoint number, starting at 0
    :return: Generates or amends the text file in the local directory, places list objects formatted as "[voltage, yav, hav]" on each line
    """
    voltage = config.start_voltage + (datapoint_num * config.voltage_increment)
    cut_file_name = config.video_file.replace('.avi', '')
    try:
        if os.stat(str(cut_file_name) + '_data.txt').st_size != 0 and frame_num <= 70:
            acknowledgement = ""
            while acknowledgement != "continue":
                acknowledgement = input(
                    'File name already contains data. Please cancel and clear the file before proceeding. Type "continue" to override')
            print("\ncontinuing...")
        with open(str(cut_file_name) + '_data.txt', 'a') as f:
            if config.pixel_to_mm == None:
                conversion = 1
            else:
                conversion = config.pixel_to_mm
            yav_mm = yav * conversion
            hav_mm = hav * conversion
            if (yav_mm, hav_mm) != (0, 0):
                f.write('[' + str(voltage) + ', ' + str(round(yav_mm, 2)) + ', ' + str(round(hav_mm, 2)) + ']\n')
                percentage = (frame_num / total_frames) * 100
                print("Saved: " + str(voltage) + ', ' + str(round(yav_mm, 2)) + ', ' + str(round(hav_mm, 2)) + '; Completion: ' + str(round(percentage, 0)) + '%, ' + str(frame_num))
            else:
                print('No Particle Detected')
    except FileNotFoundError:
        with open(str(cut_file_name) + '_data.txt', 'w') as f:
            if config.pixel_to_mm == None:
                conversion = 1
            else:
                conversion = config.pixel_to_mm
            yav_mm = yav * conversion
            hav_mm = hav * conversion
            if (yav_mm, hav_mm) != (0, 0):
                f.write('[' + str(voltage) + ', ' + str(round(yav_mm, 2)) + ', ' + str(round(hav_mm, 2)) + ']\n')
                percentage = (frame_num / total_frames) * 100
                print("Saved: " + str(voltage) + ', ' + str(round(yav_mm, 2)) + ', ' + str(round(hav_mm, 2)) + '; Completion: ' + str(round((percentage), 0)) + '%, ' + str(frame_num))
            else: 
                print('No Particle Detected')


def auto_run(cap, config = get_default_config()):
    """
    Automatic processing of video frames, outputs datapoints as described below in a text data file
    :param cap: Video capture object from the OpenCV package
    :return: Generates or amends the text file in the local directory, places list objects formatted as "[voltage, y-position, height]" on each line
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracking_objects, track_id, keypoints_prev_frame = setup_tracking()
    _, _, y_start, y_end = gen_initial_frame(cap)
    collection_frames = [
        int((config.fps * config.change_interval * i) + 
            (config.fps * config.change_interval * 0.4))
        for i in range(100)
    ]
    end_collection_frames = [cf + config.sample_frames for cf in collection_frames]
    _, datapoint = [], []
    collect_data = False
    keypoints_prev_frame = []
    datapoint_num = 0
    for frame_num in range(config.start_frame, total_frames):
        ret, frame = get_frame(cap, frame_num)
        if frame_num >= config.start_frame + 2 and len(tracking_objects.keys()) > 0 :
            for i in tracking_objects.keys():
                try:
                    last_known = tracking_objects[i]
                except KeyError:
                    pass
        tracking_objects, track_id, _ = setup_tracking()
        if not ret:
            break
        roi_frame, closing, _, _ = post_processing(cap, frame, frame_num)
        if frame_num >= config.start_frame + 2:
            x, y, h, _, keypoints_cur_frame = locate_particles(roi_frame, closing, keypoints_prev_frame, 
                                 frame_num, tracking_objects, track_id, y_end, y_start, last_known)
        else:
            x, y, h, _, keypoints_cur_frame = locate_particles(roi_frame, closing, keypoints_prev_frame, 
                                 frame_num, tracking_objects, track_id, y_end, y_start)
        keypoints_prev_frame = keypoints_cur_frame
        if frame_num in collection_frames:
            collect_data = True
        if frame_num in end_collection_frames:
            collect_data = False
            xav, yav, hav = analyze_trial(datapoint)
            save_data(yav, hav, frame_num, total_frames, datapoint_num)
            datapoint_num = datapoint_num + 1
            datapoint = []
        if collect_data and x != "NaN":
            datapoint.append([x, y, h])
        if frame_num == total_frames - 1:
            print("Video processing complete")
            break


def run_frame(cap, frame_num, keypoints_prev_frame):
    """
    Manually processes and displays each frame. Press a letter or arrow key to progress
    :param cap: Video capture object from the OpenCV package
    :param frame_num: Frame number of interest
    :param keypoints_prev_frame: List object containing tuples of particle locations in the previous frame
    :return frame_num: Frame number of the next frame to analyze
    :return keypoints_cur_frame: List object containing detected objects of current frame
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
    
    config = MicromotionTrackingConfig()

    cap = cv2.VideoCapture(config.video_file)
    _, _, _, _ = gen_initial_frame(cap)
    
    frame_num = config.start_frame
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