import cv2
import numpy as np
"""
This is where we define all of our functions for tracking particle height
in tracking.py and particle position in shuttle_tracking.py
"""

# -----------------------------------Shuttling Tracking-----------------------------------
#____________________________________Tracking Class and Subclasses________________________
class TrackingConfig:
    def __init__(self, video_file = " ", start_frame = 100, fps = 20, bin_thresh = 30, x_range = (100, 1000), y_range = (100, 1000), \
                 pixel_to_mm = 0.01628, top_rect= ((0, 0), (0, 0)), left_rect= ((0, 0), (0, 0)), right_rect= ((0, 0), (0, 0)), \
                 bottom_rect= ((0, 0), (0, 0))):
        #check
        self.video_file = video_file
        self.view_type = "image"        # "image" to block out white binary noise, "binary" to block out black binary noise
        self.start_frame = start_frame 
        self.fps = fps
        self.bin_thresh = bin_thresh
        self.x_range = x_range       # x-axis frame of interest limits
        self.y_range = y_range      # y-axis frame of interest limits
        self.top_rect= top_rect
        self.left_rect= left_rect
        self.right_rect= right_rect
        self.bottom_rect= bottom_rect
        self.rectangle_color = (0, 0, 0)
        self.pixel_to_mm = pixel_to_mm    # Pixel-to-millimeter conversion, gathered from calibration image. "None" will output raw pixel data

class MicromotionConfig(TrackingConfig):
    def __init__(self, video_file = " ", start_frame = 100, fps = 20, bin_thresh = 30, x_range = (100, 1000), y_range = (100, 1000) \
                 , pixel_to_mm = 0.01628, top_rect= ((0, 0), (0, 0)), left_rect= ((0, 0), (0, 0)), right_rect= ((0, 0), (0, 0)), \
                 bottom_rect= ((0, 0), (0, 0)), view_type = "image", start_voltage = 40, voltage_increment = 5, \
                change_interval = 5, sample_frames = 15):
        self.view_type = view_type        # "image" to block out white binary noise, "binary" to block out black binary noise
        self.start_voltage = start_voltage         # Initial voltage value 
        self.voltage_increment = voltage_increment      # Voltage step between datapoints
        self.change_interval = change_interval        # Time between data points in the real-time trial (seconds)
        self.sample_frames = sample_frames         # Number of frames averaged over per data point
        super().__init__(video_file = video_file, start_frame = start_frame, fps = fps, bin_thresh = bin_thresh, x_range = x_range, y_range = y_range \
                 , pixel_to_mm = pixel_to_mm, top_rect = top_rect, left_rect= left_rect, right_rect=right_rect, \
                 bottom_rect= bottom_rect)

class ShuttlingConfig(TrackingConfig):
     def __init__(self, video_file = " ", start_frame = 100, fps = 20, bin_thresh = 30, x_range = (100, 1000), y_range = (100, 1000) \
                 , pixel_to_mm = 0.01628, data_storage = open('_.txt', 'a'), all_indices_of_interest = [], \
                 store_height_data = False, contour_det = False, collect_position = True, image_save = True, image_save_times = [0, 2, 4, 6]):
         
         self.cleaning_kernel = np.ones((2, 2), np.uint8)
         self.filling_kernel = np.ones((4, 2), np.uint8)
         self.store_height_data = store_height_data
         self.contour_det = contour_det
         self.collect_position = collect_position
         self.all_indices_of_interest = []
         self.image_save = image_save
         self.image_save_times = image_save_times
         self.data_storage = data_storage
         super().__init__(video_file, start_frame, fps, bin_thresh, x_range, y_range \
                 , pixel_to_mm)
#________________________________________________________________________________________________________________________________________


def get_frame(cap, got_frame_num):
    """
    Reads a specific frame, converting from a cv2 video capture to an individual frame
    :param cap: cv2 video capture
    :param got_frame_num: frame number
    :return: (frame validity boolean, frame)
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, got_frame_num)
    return cap.read()

def collect_pos_data(start_x, storage_file, tracking_objects_dict, index_of_interest, all_indices_of_interest, frame_num):
    """
    Puts position data from a main loop dictionary into a file
    :param start_x: First data point's pixel x-coordinate
    :param storage_file:
    :param tracking_objects_dict: Main loop position data structure
    :param index_of_interest: identify an object to store data for
    :param all_indices_of_interest:
    :param frame_num:
    :return: None
    """
    if len(tracking_objects_dict.keys()) > 0:
        for i in all_indices_of_interest:
            if i in tracking_objects_dict.keys():
                index_of_interest = i
        if index_of_interest in tracking_objects_dict.keys():
            storage_file.write(str(frame_num) + ',' + str(tracking_objects_dict[index_of_interest][0][0] - start_x) + '\n')


# ------------------------------------Height Tracking------------------------------------


# ------------------------------------------Both------------------------------------------

def set_up_detector():
    """
    Sets up the opencv blob detector
    :param: None
    :return: detector object
    """

    params = cv2.SimpleBlobDetector.Params()

    # Filter by color
    params.filterByColor = True
    params.blobColor = 255

    # Filter by area (pixels)
    params.filterByArea = False
    params.minArea = 2
    params.maxArea = 600

    # Filter by circularity
    params.filterByCircularity = False

    # Filter by convexity
    params.filterByConvexity = False

    # Filter by inertia ratio (To detect elongated shapes)
    params.filterByInertia = False
    params.minInertiaRatio = 0.01
    params.maxInertiaRatio = 0.3

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector()
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    return detector


def post_processing(thresh, cleaning_kernel, filling_kernel, rectangle_color, top_rect_pt1, top_rect_pt2, left_rect_pt1, left_rect_pt2,
                    right_rect_pt1, right_rect_pt2, bottom_rect_pt1, bottom_rect_pt2, clean_iter, dilate_iter, close_iter):
    """
    Runs a post-processing sequence that cleans noise out of a frame
    :param thresh: Binary threshold image
    :param cleaning_kernel: Numpy matrix
    :param filling_kernel: Numpy matrix
    :param clean_iter: Number of erosion iterations for cleaning
    :param dilate_iter: Number of dilation iterations for filling holes
    :param close_iter: Number of erosion iterations for returning to original particle size
    """
    clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cleaning_kernel, iterations=clean_iter)
    cv2.rectangle(clean_thresh, top_rect_pt1, top_rect_pt2, rectangle_color, -1)  # Top Erasure
    cv2.rectangle(clean_thresh, left_rect_pt1, left_rect_pt2, rectangle_color, -1)   # Left Erasure
    cv2.rectangle(clean_thresh, right_rect_pt1, right_rect_pt2, rectangle_color, -1)  # Right Erasure
    cv2.rectangle(clean_thresh, bottom_rect_pt1, bottom_rect_pt2, rectangle_color, -1)  # Bottom erasure
    dilation = cv2.dilate(clean_thresh, filling_kernel, iterations=dilate_iter)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, filling_kernel, iterations=close_iter)
    return clean_thresh, closing


def save_image(name, time, image_save_times, frame):
    """
    Saves images at specified times
    :param name: File name
    :param image_save_times: List of times(s) to save an image at
    """
    if time in image_save_times:
        # Save the frame as an image file
        cv2.imwrite(name + str(time) + '.tif', frame)

def setup_tracking():
    """
    Initializing tracking variables
    """
    return {}, 0, []