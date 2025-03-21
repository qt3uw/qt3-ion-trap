import cv2
import numpy as np
import math
from tracking_methods import collect_pos_data, set_up_detector, get_frame, post_processing, save_image
from uncertainties import Uncertainties
# --------------------------- Config ---------------------------------------------- #


class ShuttleTrackingConfig:
    """
   ShuttleTrackingConfig defines the multitude of parameters used to track, log, analyze, and graph shuttling/splitting data from the raw .avi files
        :param video_path: (str) The directory for the .avi files that one wants to analyze
        :param storage_path: (str) The directory for writing and storing the shuttle/splitting data
        :param x_bounds: (List[int int]) The bounds of the x-region with which we want to analyze particle motion
        :param y_bounds: (List[int int]) The bounds of the y-region with which we want to analyze particle motion
        :param rects: (List[Tuple((int, int), (int, int)), Tuple((int, int), (int, int)), Tuple((int, int), (int, int)), Tuple((int, int), (int, int))])
                       The top-left corner and bottom-right corner of rectangles which are selected to be ignored in the tracking and data analysis
        :param split: (bool) Indicates if it is a shuttling or splitting experiement
        :field start_frame: (int) The definition for the t=0 frame
        :field x_start: (int) x_bounds[0]
        :field y_start: (int) y_bounds[0]
        :field x_end: (int) x_bounds[1]
        :field y_end: (int) y_bounds[1]
        :field bin_thresh: (int) Threshold value for whether a pixel region is bright enough to be counted as a particle. The higher the bin_thresh, the more
                           bright the region needs to be
        :field cleaning_kernel: np.ones((2, 2), np.uint8)
        :field filling_kernel: np.ones((4, 2), np.uint8)
        :field top_rect: (Tuple(Tuple(int, int), Tuple(int, int))) rects[0]
        :field left_rect: (Tuple(Tuple(int, int), Tuple(int, int))) rects[3]
        :field right_rect: (Tuple(Tuple(int, int), Tuple(int, int))) rets[1]
        :field bottom_rect: (Tuple(Tuple(int, int), Tuple(int, int))) rects[2]
        :field rectangle_color: (Tuple(int, int int)) The color of the rectangle which indicates ignored regions in the images
        :field rects: (List[Tuple((int, int), (int, int)), Tuple((int, int), (int, int)), Tuple((int, int), (int, int)), Tuple((int, int), (int, int))])
                      [top_rect, right_rect, bottom_rect, left_rect]
        :field store_height_data: (bool) Indicates if the user wants to store the particle height data
        :field contour_det: (bool) Toggles contour on and off
        :field collect_position: (bool) Toggles the option of savinf particle position or not
        :field all_indices_of_interest: (List[int]) Indicates specific indices which will be counted 
                                        in the data collection/analysis
        :field image_save: (bool) Toggles saving a sample of images from the particle motion
        :field image_save_times: (List[float]) Indicates what times in seconds in the .avi the user wants to save a frame 
        :field data_storage: (str) open(storage_path, 'w')
        :field split: (bool) Indicates if it is a shuttling or splitting experiement
    """
    def __init__(self, video_path = 'E:/Feb 27-28 Experimental Data Collection/02-28-2025 Shuttle-Split/Clean Data/02-28-2025_Trial_split.avi', storage_path = 'data/shuttling/02-28-2025_split_data.txt', \
                 x_bounds = [0,1350], y_bounds = [650, 800], rects = [((0, 0), (0, 0)), ((0, 0), (0, 0)), ((0, 0), (0, 0)), ((0, 0), (0, 0))], split = False):
        # video settings
        self.video_path = video_path
        self.start_frame_num = 10
        # regions of interest
        self.x_start = int(x_bounds[0])
        self.y_start = int(y_bounds[0])
        self.x_end = int(x_bounds[1])
        self.y_end = int(y_bounds[1])
        # image processing
        self.bin_thresh = 45
        self.cleaning_kernel = np.ones((2, 2), np.uint8)
        self.filling_kernel = np.ones((4, 2), np.uint8)
        # frame erasure rectangles
        self.top_rect = rects[0]
        self.left_rect = rects[3]
        self.right_rect = rects[1]
        self.bottom_rect = rects[2]
        self.rectangle_color = (0 ,0, 50)
        self.rects = [self.top_rect, self.right_rect, self.bottom_rect, self.left_rect]
        # tracking settings
        self.store_height_data = False
        self.contour_det = True
        self.collect_position = True
        self.all_indices_of_interest = [i for i in range(0, 300)]
        # image capture settings
        self.image_save = True
        self.image_save_times = [0, .25, .5, .75, 1, 1.25]
        # data storage
        self.data_storage = open(storage_path, 'w')
        self.split = split
    

# --------------------------- Video Processing Functions ---------------------------------------------- #


def initialize_video(config):
    """
    Turns .avi video into a format recognizable to OpenCv. 
        :returns:
            :cap: (cv2.VideoCapture()) OpenCv object using the .avi file at config.video_path
            :total_frames: (int) The total amount of video frames in the .avi file
            :start_frame: (int) The frame which the user wants to mark as the t=0 frame
    """
    cap = cv2.VideoCapture(config.video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, start_frame = get_frame(cap, config.start_frame_num)
    if ret:
        frame_height, frame_width = start_frame.shape[:2]
        print(f"Frame height: {frame_height}\nFrame width: {frame_width}")
        roi = start_frame[config.y_start:config.y_end, config.x_start:config.x_end]
        cv2.imshow("Frame", roi)
    return cap, total_frames, start_frame

# --------------------------- Setup Functions For Tracking ---------------------------------------------- #


def setup_tracking():
    """
    Establishes empty values for 'tracking_objects,' 'track_id,' and 'key_points_prev_frame'
    :return: {}, 0, []
    """
    return {}, 0, []  # tracking_objects, track_id, keypoints_prev_frame


def process_frame(config, frame):
    """
    Processes a given frame
        :param config: (ShuttleTrackingConfig) A ShuttleTrackingConfig object
        :param frame: The specific frame that is being processed
        :returns:
            :roi_frame: region of interest frame
            :closing:
            :clean_thresh:
    """
    roi_frame = frame[config.y_start:config.y_end, config.x_start:config.x_end]
    gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_frame, config.bin_thresh, 255, cv2.THRESH_BINARY)
    clean_thresh, closing = post_processing(
        thresh,
        config.cleaning_kernel,
        config.filling_kernel,
        config.top_rect[0], config.top_rect[1],
        config.left_rect[0], config.left_rect[1],
        config.right_rect[0], config.right_rect[1],
        config.bottom_rect[0], config.bottom_rect[1],
        0, 1, 4
    )
    return roi_frame, closing, clean_thresh


def update_tracking(tracking_objects, track_id, keypoints_cur_frame, keypoints_prev_frame, frame_num, contours=None):
    """
    Updates the information about what object we are tracking by keeping track propertys related to the indices of 
    interest
        :param tracking_objects: (Dict{int : Point()}) Associates a track_id int with an OpenCv point object for
                                 a detected particle.
        :param track_id: (int) The keyword used for tracking_objects
        :param keypoints_cur_frame: (Point()) OpenCv point object of detected particle in the current frame
        :param key_points_prev_frame: (Point()) OpenCv point object of detected particle in the previous frame
        :param frame_num: (int) The frame number of the current frame 
        :param contours: (Countour()) Countour object of the detected particle that is used to create a bounding 
                         box around the particle
        :returns:
            :tracking objects: (Dict{int : Point()} Associates a track_id int with an OpenCv point object for
                                 a detected particle.
            :track_id: (int) the current tracking index of the particle
    """
    if frame_num <= 2:
        for pt1 in keypoints_cur_frame:
            for pt2 in keypoints_prev_frame: 
                if math.dist(pt1, pt2) < 10:
                    tracking_objects[track_id] = [pt1]
                    track_id += 1
    else:
        tracking_objects_copy = tracking_objects.copy()
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
    for pt1 in keypoints_cur_frame:
        tracking_objects[track_id] = [pt1]
        track_id += 1
    if contours is not None:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            for key in tracking_objects.keys():
                if x <= tracking_objects[key][0][0] <= x + w and y <= tracking_objects[key][0][1] <= y + h:
                    tracking_objects[key].append(h)
    return tracking_objects, track_id


def draw_frame_info(image, frame_num, time, total_frames):
    """
    Draws the frame number and corresponding time in seconds that have passed since 
    the user-defined t=0 frame.
        :param image: (image) The frame with which the user wants to overlay with
                      tracking information in real time.
        :param frame_num: (int) The frame number of the image we are overlaying
        :param time: (float) The time in seconds that has elapsed since the user-defined t=0
                     frame
        :param total_frames: (int) The amount of frames in the entire analyzed video
    """
    if frame_num >= total_frames - 1:
        cv2.putText(image, "Frame: end", (5, 20), 0, 0.5, (0, 255, 0), 1)
    else:
        cv2.putText(image, f"Frame: {frame_num}", (5, 20), 0, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"Time: {time}", (5, 40), 0, 0.5, (255, 255, 255), 1)


def draw_tracking_info(image, tracking_objects):
    """
    Draws trac object_id and object position on a video file in real time
        :param image: (image) 
        :param tracking_objects: (Dict{int, Point()}) A dictionary that associates
                                 a tracking_id int with an OpenCv point object.
    """
    for object_id, item in tracking_objects.items():
        cv2.putText(image, str(object_id),
                   (int(item[0][0] - 5), int(item[0][1] - 17)),
                   0, 0.5, (0, 255, 0), 1)

# --------------------------- Main Processing Loop ---------------------------------------------- #


def run_tracking(config, cap, detector, total_frames, start_frame, origin_part = False, key0 = None):
    """
    Automaticallly analyzes and saves the video data for shuttling/splitting 
        :param config: (ShuttleTrackingConfig) ShuttleTrackingConfig object 
        :param cap: (VideoCapture) an OpenCv VideoCapture() object
        :param detector: (SimpleBlobDetector) An OpenCv SimpleBlobDetector object
        :param total_frames: (int) The total amount of frames in the video data
        :param start_frame: (int) The frame the user defines to be at t=0
        :param origin_part: (bool) Toggles where the starting position is defined
        :param key0: (int) ASCII number for user key press
    """
    frame_num = config.start_frame_num
    tracking_objects, track_id, keypoints_prev_frame = setup_tracking()
    # Initial tracking variables
    index_of_interest = 0
    first_detect = False
    start_x = 0     
    run = True
    run_body = True      
    key = 0
    while run:
        frames_to_play = 0
        if key0 != None:
            key = key0
        else:
            key = cv2.waitKey()
        if key == 27:  # ESC
            run = False
        elif key == 32:  # Space
            frames_to_play = 20
        elif key == 39:
            frames_to_play = 1
        elif key == 38:
            frames_to_play = total_frames
        for _ in range(frames_to_play):
            if total_frames <= frame_num and key == 38:
                break
            ret, frame = get_frame(cap, frame_num)
            if not ret and frame_num < total_frames:
                print("Cannot retrieve frame.")
                run = False
                break
            elif frame_num >= total_frames:
                if cv2.waitKey() == 27:  # ESC
                    run = False
                    run_body = False
                break
            if not run_body:
                break
            # process frame
            roi_frame, closing, clean_thresh = process_frame(config, frame)
            # detect particles
            keypoints = detector.detect(closing)
            keypoints_cur_frame = [kp.pt for kp in keypoints]
            image_with_keypoints = cv2.drawKeypoints(roi_frame, keypoints, np.array([]), (0, 0, 255))
            # Find contours if enabled
            contours = None
            if config.contour_det:
                contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Update tracking
            tracking_objects, track_id = update_tracking(
                tracking_objects, track_id, keypoints_cur_frame, 
                keypoints_prev_frame, frame_num, contours
            )
            # set starting position
            if not first_detect and origin_part and len(tracking_objects.keys()) > index_of_interest:
                start_x = tracking_objects[index_of_interest][0][0]
                print("Start x:", start_x)
                first_detect = True
            # collect position data if enabled
            if config.collect_position:
                collect_pos_data(
                    start_x,
                    config.data_storage,
                    tracking_objects,
                    index_of_interest,
                    config.all_indices_of_interest,
                    frame_num
                )
            # draw information
            draw_tracking_info(image_with_keypoints, tracking_objects)
            time = round((frame_num - 100) * (1.0/60.0), 2)
            draw_frame_info(clean_thresh, frame_num, time, total_frames)
            [cv2.rectangle(image_with_keypoints, rect[0], rect[1], config.rectangle_color, -1) for rect in config.rects]
            draw_frame_info(image_with_keypoints, frame_num, time, total_frames)
            # if enabled, save image
            if config.image_save:
                if config.split:
                    save_image('figures/figure_5/shuttle_images/NewShuttleParticleAtTime', time,
                                config.image_save_times, clean_thresh)
                else:
                    save_image('figures/figure_5/split_images/NewParticleAtTime', time,
                                config.image_save_times, clean_thresh)
            # Display frame
            cv2.imshow("Frame", image_with_keypoints)
            cv2.waitKey(50)
            keypoints_prev_frame = keypoints_cur_frame
            frame_num += 1
        if total_frames <= frame_num:
            break

def split_data(config, cap, detector, total_frames, start_frame):
    """
    Analyzes and saves positional data for a two-particle split scenerio
        :param config: (ShuttleTrackingConfig) ShuttleTrackingConfig object 
        :param cap: (VideoCapture) an OpenCv VideoCapture() object
        :param detector: (SimpleBlobDetector) An OpenCv SimpleBlobDetector object
        :param total_frames: (int) The total amount of frames in the video data
        :parram start_frame: (int) The frame the user defines to be at t=0
    """
    config_0 =  config
    ion_xs = []
    center_x = 0
    key = cv2.waitKey()
    run_tracking(config, cap, detector, total_frames, start_frame, key0 = key)
    with open('data/split/02-28-2025_split_data.txt', 'r') as file:
        for line in file:
            tuple_str = line.strip()
            tuple_data = eval(tuple_str)
            ion_xs.append(tuple_data[1])
        center_x = np.abs(np.mean(ion_xs[0:1])) + np.abs(ion_xs[0]) 
    l_domain = np.ones(2) * config.x_start + np.array([0,  675])
    r_domain = np.ones(2) * l_domain[1] + np.array([0, 675])
    i = 0
    l_r = ['data/split/02-28-2025_left_split_data.txt', 'data/split/02-28-2025_right_split_data.txt']
    if key != 38:
        key = cv2.waitKey()
    #rects_l = [((0, 0), (0, 0)), ((l_domain[1], 0)), (r_domain[1], config_0.y_end - config_0.y_start),  ((0, 0), (0, 0)), ((0, 0), (0, 0))]
    config_l = ShuttleTrackingConfig(storage_path = l_r[0], rects = [((0, 0), (0, 0)), ((int(l_domain[1]), 0), (int(r_domain[1]), int(config_0.y_end - config_0.y_start))),  ((0, 0), (0, 0)), ((0, 0), (0, 0))])
    initialize_video(config_l)
    run_tracking(config_l, cap, detector, total_frames, start_frame, key0 = key)
    if key != 38:
        key = cv2.waitKey()
    config_r = ShuttleTrackingConfig(storage_path = l_r[1], rects = [((0, 0), (0, 0)), ((0, 0), (0, 0)), ((0, 0), (0, 0)), ((0, 0), (int(l_domain[1]), int(config_0.y_end - config_0.y_start)))])
    initialize_video(config_r)
    run_tracking(config_r, cap, detector, total_frames, start_frame, key0 = key)


    
    


def main():
    """
    Automatically analyzes and saves data for the shuttling and splitting video data. 
    """
    print("Running program...")
    config_shuttle = ShuttleTrackingConfig(video_path ='E:/Feb 27-28 Experimental Data Collection/02-28-2025 Shuttle-Split/Clean Data/02-28-2025_Trial_split.avi', storage_path = 'data/split/02-28-2025_split_data.txt')
    cap, total_frames, start_frame = initialize_video(config_shuttle)
    detector_shuttle = set_up_detector()
    #run_tracking(config, cap, detector, total_frames, start_frame)
    split_data(config_shuttle, cap, detector_shuttle, total_frames, start_frame)
    config_shuttle =  ShuttleTrackingConfig(video_path ='E:/Feb 27-28 Experimental Data Collection/02-28-2025 Shuttle-Split/Clean Data/02-28-2025_Trial_shuttle.avi', storage_path = 'data/shuttling/02-28-2025_shuttle_data.txt')
    cap, total_frames, start_frame = initialize_video(config_shuttle)
    detector_shuttle = set_up_detector()
    run_tracking(config_shuttle, cap, detector_shuttle, total_frames, start_frame)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    