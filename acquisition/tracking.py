import os
import cv2
import numpy as np
import math

class TrackingConfig:
    def __init__(self):
        self.VIDEO_FILE = "acquisition/8-16Trial4.avi"
        self.VIEW_TYPE = "image"  # Options: "binary", "image", "frame", "cleanthresh"
        self.OUTPUT = "none"      # Options: "tuple", "none"
        self.SHOW_QUANT = "none"  # Options: "height", "micromotion", "both"
        self.POOR_TRACKING = False
        self.FPS = 20
        self.CHANGE_INTERVAL = 5
        self.POINTS_TO_IMAGE = []
        self.SAMPLE_FRAMES = 15
        self.BIN_THRESH = 26
        self.X_RANGE = (800, 1200)
        self.Y_RANGE = (547, 933)
        self.BOTTOM_BAR = 100
        self.TOP_BAR = 0
        self.LEFT_BAR = 0
        self.RIGHT_BAR = 0
        self.INDEX_LIST = []
        self.TRACKING_OBJECTS = {}

def initialize_video(cap):
    """Initialize video capture and kernels"""
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cleaning_kernel = np.ones((2, 2), np.uint8)
    filling_kernel = np.ones((2, 2), np.uint8)
    return total_frames, cleaning_kernel, filling_kernel

def get_frame(cap, frame_num):
    """Get a specific frame from video"""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    return cap.read()

def frame_dimensions(cap, frame_num):
    """Calculate frame dimensions and ranges"""
    ret, start_frame = get_frame(cap, frame_num)
    start_frame_dim = start_frame.shape
    imageheight = start_frame_dim[0]
    x_start, x_end = config.X_RANGE
    y_start, y_end = (imageheight - config.Y_RANGE[1]), (imageheight - config.Y_RANGE[0])
    return x_start, x_end, y_start, y_end

def gen_initial_frame(cap):
    """Generate and display initial frame"""
    total_frames, cleaning_kernel, filling_kernel = initialize_video(cap)
    x_start, x_end, y_start, y_end = frame_dimensions(cap, 1)
    ret, start_frame = get_frame(cap, 1)
    cv2.imshow("Frame", start_frame[y_start:y_end, x_start:x_end])
    return x_start, x_end, y_start, y_end

def setup_detector():
    """Configure and create blob detector"""
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
    return cv2.SimpleBlobDetector_create(params)

def define_blockers(cap, frame_num):
    """Define blocking rectangles for frame processing"""
    x_start, x_end, y_start, y_end = frame_dimensions(cap, frame_num)
    ylength = y_end - y_start
    xlength = x_end - x_start
    
    top_rect = ((0, 0), (1616, config.TOP_BAR))
    left_rect = ((0, 0), (config.LEFT_BAR, 1240))
    right_rect = ((xlength - config.RIGHT_BAR, 0), (xlength, 1240))
    bottom_rect = ((0, ylength - config.BOTTOM_BAR), (1616, ylength))
    
    return (*top_rect, *left_rect, *right_rect, *bottom_rect)

def post_processing(cap, frame, frame_num):
    """Process frame and apply filters"""
    x_start, x_end, y_start, y_end = frame_dimensions(cap, frame_num)
    blockers = define_blockers(cap, frame_num)
    rectangle_color = (255, 255, 255) if config.VIEW_TYPE == "binary" else (0, 0, 0)
    
    # intialize kernels
    cleaning_kernel = np.ones((2, 2), np.uint8)
    filling_kernel = np.ones((2, 2), np.uint8)
    
    # process image
    roi_frame = frame[y_start:y_end, x_start:x_end]
    gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_frame, config.BIN_THRESH, 255, cv2.THRESH_BINARY)
    
    # apply morphological operations
    clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cleaning_kernel, iterations=1)
    
    # apply blocking rectangles
    for i in range(0, len(blockers), 2):
        cv2.rectangle(clean_thresh, blockers[i], blockers[i+1], rectangle_color, -1)
    
    closing = cv2.morphologyEx(clean_thresh, cv2.MORPH_CLOSE, filling_kernel, iterations=2)
    return roi_frame, closing, clean_thresh

def setup_tracker():
    """Initialize tracking objects"""
    return {}, 0, []

def locate_particles(roi_frame, closing, keypoints_prev_frame, frame_num, tracking_objects, track_id, y_end, y_start):
    """Locate and track particles in frame"""
    detector = setup_detector()
    keypoints = detector.detect(closing)
    keypoints_cur_frame = []
    x_position, y_position, height = "NaN", "NaN", "NaN"
    
    # extract keypoints
    for keypoint in keypoints:
        keypoints_cur_frame.append(keypoint.pt)

    
    image_with_keypoints = cv2.drawKeypoints(roi_frame, keypoints, np.array([]), (0, 0, 255))
    
    # find contours
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # track particles
    if frame_num <= 2:
        track_id = _initialize_tracking(keypoints_cur_frame, keypoints_prev_frame, tracking_objects, track_id)
    else:
        _update_tracking(keypoints_cur_frame, tracking_objects)
    
    # process contours and get measurements
    _process_contours(contours, tracking_objects, y_end, y_start)
    
    # get position data
    if frame_num >= 2 and len(tracking_objects.keys()) > 0:
        try:
            x_position = int(tracking_objects[0][0][0])
            y_position = int(tracking_objects[0][0][1])
            height = int(tracking_objects[0][1])
        except (KeyError, IndexError):
            pass
    
    return x_position, y_position, height, image_with_keypoints, keypoints_cur_frame

def _initialize_tracking(keypoints_cur_frame, keypoints_prev_frame, tracking_objects, track_id):
    """Initialize tracking for new particles"""
    for pt1 in keypoints_cur_frame:
        for pt2 in keypoints_prev_frame:
            if math.dist(pt1, pt2) < 10:
                tracking_objects[track_id] = [pt1]
                track_id += 1
    return track_id

def _update_tracking(keypoints_cur_frame, tracking_objects):
    """Update tracking for existing particles"""
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

    track_id_2 = 0
    for pt1 in keypoints_cur_frame:
        tracking_objects[track_id_2] = [pt1]
        track_id_2 += 1


def _process_contours(contours, tracking_objects, y_end, y_start):
    """Process contours to get particle measurements"""
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        centroid_x = int(x + w / 2)
        centroid_y = int(y + h / 2)
        adj_centroid_y = (y_end - y_start) - centroid_y
        
        for key in tracking_objects.keys():
            if (x <= tracking_objects[key][0][0] <= x + w and 
                y <= tracking_objects[key][0][1] <= y + h):
                tracking_objects[key].append(h)

def analyze_trial(datapoint):
    """Analyze trial data and compute averages"""
    if not datapoint:
        return 0, 0, 0
        
    x = [point[0] for point in datapoint]
    y = [point[1] for point in datapoint]
    h = [point[2] for point in datapoint]
    
    return (round(np.mean(x), 2),
            round(np.mean(y), 2),
            round(np.mean(h), 2))

def save_data(yav, hav, y_start, y_end, frame_num):
    try:
        if os.stat('Tuple.txt').st_size != 0 and frame_num < 150:
            acknowledgement = ""
            while acknowledgement != "continue":
                acknowledgement = input(
                    'Tuple.txt already contains data. Type "continue" to add to the existing file, otherwise stop. ')
            print("\ncontinuing...")
        with open('Tuple.txt', 'a') as f:
            yav_oriented = (y_end - y_start) - yav
            f.write('[' + str(round(yav_oriented, 2)) + ', ' + str(round(hav, 2)) + ']\n')
            print("Saved: " + str(yav_oriented) + ', ' + str(hav))
    except FileNotFoundError:
        with open('Tuple.txt', 'w') as f:
            yav_oriented = (y_end - y_start) - yav
            f.write('[' + str(round(yav_oriented, 2)) + ', ' + str(round(hav, 2)) + ']\n')
            print("Saved: " + str(yav_oriented) + ', ' + str(hav))

def auto_run(cap):
    """Automatic processing of video frames"""
    total_frames, _, _ = initialize_video(cap)
    tracking_objects, track_id, keypoints_prev_frame = setup_tracker()
    x_start, x_end, y_start, y_end = gen_initial_frame(cap)
    
    # calculate collection frames
    collection_frames = [
        int((config.FPS * config.CHANGE_INTERVAL * i) + 
            (config.FPS * config.CHANGE_INTERVAL * 0.4))
        for i in range(100)
    ]
    end_collection_frames = [cf + config.SAMPLE_FRAMES for cf in collection_frames]
    
    trial, datapoint = [], []
    collect_data = False
    
    # process frames
    for frame_num in range(total_frames):
        ret, frame = get_frame(cap, frame_num)
        if not ret:
            break
        if frame_num == 0:
            keypoints_passover = []
        roi_frame, closing, clean_thresh = post_processing(cap, frame, frame_num)
        x, y, h, dummyvar, keypoints_prev_frame = locate_particles(roi_frame, closing, keypoints_passover, 
                                 frame_num, tracking_objects, track_id, y_end, y_start)
        keypoint_passover = keypoints_prev_frame
        
        # collect and analyze data
        if frame_num in collection_frames:
            print('collection started')
            collect_data = True
        if frame_num in end_collection_frames:
            print('collection ended')
            collect_data = False
            xav, yav, hav = analyze_trial(datapoint)
            trial.append([xav, yav, hav])
            save_data(yav, hav, y_start, y_end, frame_num)
            print(trial)
            datapoint = []
        if collect_data and x != "NaN":
            datapoint.append([x, y, h])

def run_frame(cap, frame_num, keypoints_prev_frame):
    tracking_objects, track_id, _ = setup_tracker()
    ret, frame = get_frame(cap, frame_num)
    if not ret:
        exit()
    x_start, x_end, y_start, y_end = gen_initial_frame(cap)
    roi_frame, closing, clean_thresh = post_processing(cap, frame, frame_num)
    x, y, h, image_with_keypoints, keypoints_cur_frame = locate_particles(roi_frame, closing, keypoints_prev_frame, 
                                frame_num, tracking_objects, track_id, y_end, y_start)


    cv2.imshow("Frame", image_with_keypoints)

    frame_num = frame_num + 1
    return frame_num, keypoints_cur_frame
        


def main():
    """Main entry point"""
    global config
    config = TrackingConfig()
    
    cap = cv2.VideoCapture(config.VIDEO_FILE)
    x_start, x_end, y_start, y_end = gen_initial_frame(cap)
    
    key = cv2.waitKey()
    if key == 27:  # ESC
        return
    if key == 32:  # Space
        auto_run(cap)
    else:
        cv2.destroyAllWindows()
        frame_num = 0
        for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            if i == 0:
                keypoints_prev_frame = []
            frame_num, keypoints_prev_frame = run_frame(cap, frame_num, keypoints_prev_frame)
            key = cv2.waitKey()
            if key == 27:
                exit()
            if key != 27:
                pass

if __name__ == "__main__":
    main()