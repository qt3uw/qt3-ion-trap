import cv2

"""
This is where we define all of our functions for tracking particle height
in TRACKING.py and particle position in ShuttleTracking.py
"""

# -----------------------------------Shuttling Tracking-----------------------------------

"""
Reads a specific frame, converting from a cv2 video capture to an individual frame
:param cap: cv2 video capture
:param got_frame_num: frame number
:return: (frame validity boolean, frame)
"""
def get_frame(cap, got_frame_num):
    cap.set(cv2.CAP_PROP_POS_FRAMES, got_frame_num)
    _, got_frame = cap.read()
    return _, got_frame

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
def collect_pos_data(start_x, storage_file, tracking_objects_dict, index_of_interest, all_indices_of_interest, frame_num):

    # Putting position points in a file
    if len(tracking_objects_dict.keys()) > 0:
        for i in all_indices_of_interest:
            if i in tracking_objects_dict.keys():
                index_of_interest = i
        if index_of_interest in tracking_objects_dict.keys():
            storage_file.write(str(frame_num) + ',' + str(tracking_objects_dict[index_of_interest][0][0] - start_x) + '\n')


# ------------------------------------Height Tracking------------------------------------


# ------------------------------------------Both------------------------------------------

"""
Sets up the opencv blob detector
:param: None
:return: detector object
"""
def set_up_detector():
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

"""
Runs a post-processing sequence that cleans noise out of a frame
:param thresh: Binary threshold image
:param cleaning_kernel: Numpy matrix
:param filling_kernel: Numpy matrix
:param clean_iter: Number of erosion iterations for cleaning
:param dilate_iter: Number of dilation iterations for filling holes
:param close_iter: Number of erosion iterations for returning to original particle size
"""
def post_processing(thresh, cleaning_kernel, filling_kernel, rectangle_color, top_rect_pt1, top_rect_pt2, left_rect_pt1, left_rect_pt2,
                    right_rect_pt1, right_rect_pt2, bottom_rect_pt1, bottom_rect_pt2, clean_iter, dilate_iter, close_iter):
    clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cleaning_kernel, iterations=clean_iter)
    cv2.rectangle(clean_thresh, top_rect_pt1, top_rect_pt2, rectangle_color, -1)  # Top Erasure
    cv2.rectangle(clean_thresh, left_rect_pt1, left_rect_pt2, rectangle_color, -1)   # Left Erasure
    cv2.rectangle(clean_thresh, right_rect_pt1, right_rect_pt2, rectangle_color, -1)  # Right Erasure
    cv2.rectangle(clean_thresh, bottom_rect_pt1, bottom_rect_pt2, rectangle_color, -1)  # Bottom erasure
    dilation = cv2.dilate(clean_thresh, filling_kernel, iterations=dilate_iter)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, filling_kernel, iterations=close_iter)
    return clean_thresh, closing

"""
Saves images at specified times
:param name: File name
:param image_save_times: List of times(s) to save an image at
"""
def save_image(name, time, image_save_times, frame):
    if time in image_save_times:
        # Save the frame as an image file
        cv2.imwrite(name + str(time) + '.tif', frame)
