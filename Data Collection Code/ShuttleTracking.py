import cv2
import numpy as np
import math
from TrackingFunctions import collect_pos_data, set_up_detector, get_frame, post_processing, save_image

print("Running program...")

# Setting program parameters
# ------------------------------------------------------------------------------------------------------------------------

# Choosing the video to track in
VID_CAP = cv2.VideoCapture('GoodSplit.avi')

# Define region of interest (Crop the video)
X_START, Y_START = 0, 550 # Pixels
X_END, Y_END = 1600, 700 # Pixels

# Used when you need to add more indeces to the list (this happens if tracking has to jump up in indeces)
ALL_INDICES_OF_INTEREST = []

# Defining up morphological transformations kernels (Can be different for some optimizations)
CLEANING_KERNEL = np.ones((2,2), np.uint8)
FILLING_KERNEL = np.ones((4, 2), np.uint8)

# Set the binary threshold (Intensity cutoff between white and black)
BIN_THRESH = 25

# Frame erasure (Setting up black rectangles to erase unwanted regions of light)
TOP_RECT_PT1, TOP_RECT_PT2 = (0, 0), (0, 0)
LEFT_RECT_PT1, LEFT_RECT_PT2 = (0, 0), (0, 0)
RIGHT_RECT_PT1, RIGHT_RECT_PT2 = (0, 0), (0, 0)
BOTTOM_RECT_PT1, BOTTOM_RECT_PT2 = (0, 0), (0, 0)

# ADD IN ERASURE RECTANGLES TO REMOVE NOISE

# (255, 255, 255) to help place rectangles
# (0, 0, 0) to erase unwanted light
RECTANGLE_COLOR = (0, 0, 0)

# Define a starting frame number
START_FRAME_NUM = 100

# Toggle data collection
STORE_HEIGHT_DATA = False

# Toggle contour detection
CONTOUR_DET = False

# Toggle frame capture
IMAGE_SAVE = False
IMAGE_SAVE_TIMES = [0, 2, 4, 6]

# Assign a storage file for data collection
DATA_STORAGE = open('data.txt', 'a')

# Toggle data collection
COLLECT_POSITION = False
# ------------------------------------------------------------------------------------------------------------------------


# Setting up the detector
detector = set_up_detector()


# Priming the main loop
# ------------------------------------------------------------------------------------------------------------------------

# Setting up memory between frames for objects
keypoints_prev_frame = []
tracking_objects = {}
track_id = 0

# Frame counter
frame_num = START_FRAME_NUM
total_frames = int(VID_CAP.get(cv2.CAP_PROP_FRAME_COUNT))

# Trial counter
trial_num = 1
first_point = False

# Starting point (For clean graphing)
start_x = 0
first_detect = False
# ------------------------------------------------------------------------------------------------------------------------



# Main Loop
# ------------------------------------------------------------------------------------------------------------------------
ret, start_frame = get_frame(VID_CAP, frame_num)

# Object index of interest (Selecting a particle to collect data on)
index_of_interest = 1
start_frame_dim = start_frame.shape
start_frame_height = start_frame_dim[0]
start_frame_width = start_frame_dim[1]
print(f"Frame height: {start_frame_height}\n"
      f"Frame width: {start_frame_width}")

cv2.imshow("Frame", start_frame[Y_START:Y_END, X_START:X_END])

run = True
run_body = True
while run:

    # Initializing the frames for the loop to play
    frames_to_play = 0

    # Waiting for the esc key to end the loop (DEC: 27 on ASCII table)
    # Space plays 20 frames (DEC: 32 on ASCII table)
    # Any other key moves to the next frame
    key = cv2.waitKey()
    if key == 27:
        run = False
    elif key == 32:
        frames_to_play = 20
        first_point = True
    else:
        frames_to_play = 1

    # Running through the specified number of frames
    for i in range(frames_to_play):
        ret, frame = get_frame(VID_CAP, frame_num)
        if not ret and frame_num < total_frames:
            print("Cannot retrieve frame.")
            run = False
        elif frame_num >= total_frames:
            sit = True
            while sit:
                end_control = cv2.waitKey()
                if end_control == 27:
                    run_body = False
                    run = False
                    sit = False

        # Handle edge case: Running anything after the last frame causes an unnecessary error
        if not run_body:
            break

        # Frame information (Useful for debugging)
        dimensions = frame.shape
        height = dimensions[0]
        width = dimensions[1]

        # Creating the region of interest frame
        roi_frame = frame[Y_START:Y_END, X_START:X_END]
        roi_height = roi_frame.shape[0]
        roi_width = roi_frame.shape[1]

        # Converting to black and white
        gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding (Converting to intensity values of 0 and 255)
        ret, thresh = cv2.threshold(gray_frame, BIN_THRESH, 255, cv2.THRESH_BINARY)

        # Post-processing
        clean_thresh, closing = post_processing(thresh, CLEANING_KERNEL, FILLING_KERNEL, RECTANGLE_COLOR, TOP_RECT_PT1, TOP_RECT_PT2,
                        LEFT_RECT_PT1, LEFT_RECT_PT2, RIGHT_RECT_PT1, RIGHT_RECT_PT2, BOTTOM_RECT_PT1, BOTTOM_RECT_PT2,
                        0, 1, 4)

        # Finding the locations of the particles
        keypoints = detector.detect(closing)

        keypoints_cur_frame = []
        for keypoint in keypoints:
            point = keypoint.pt
            keypoints_cur_frame.append(keypoint.pt)

        # If the outimage parameter receives an argument of None or an empty array,
        # it will be a copy of the source image
        image_with_keypoints = cv2.drawKeypoints(roi_frame, keypoints, np.array([]), (0, 0, 255))

        # Finding the contours
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Compare prev and cur point only at the beginning
        if frame_num <= 2:
            for pt1 in keypoints_cur_frame:
                for pt2 in keypoints_prev_frame:
                    distance = math.dist(pt1, pt2)

                    # Add an identified object to the tracking_objects dict and prepare the next id
                    if distance < 10:
                        tracking_objects[track_id] = [pt1]
                        track_id += 1
        else:
            # Create a copy because we can't remove dictionary elements while traversing it
            tracking_objects_copy = tracking_objects.copy()
            keypoints_cur_frame_copy = keypoints_cur_frame.copy()

            for object_id, item2 in tracking_objects_copy.items():
                object_exists = False
                for pt1 in keypoints_cur_frame:
                    distance = math.dist(pt1, item2[0])

                    # Update object position
                    if distance < 10:
                        tracking_objects[object_id] = [pt1]
                        object_exists = True
                        if pt1 in keypoints_cur_frame:
                            keypoints_cur_frame.remove(pt1)
                        continue # Moves on to checking next id once we confirm existence

                # Remove lost IDs
                if not object_exists:
                    tracking_objects.pop(object_id)

        # Add found IDs
        for pt1 in keypoints_cur_frame:
            tracking_objects[track_id] = [pt1]
            track_id += 1


        if CONTOUR_DET:
            # Iterate over each contour to find the bounding rectangle and get the height
            for contour in contours:
                # Get the bounding rectangle for each contour
                x, y, w, h = cv2.boundingRect(contour)

                # CONTINUE HERE
                # Check which tracking object is in the bounding box
                for key in tracking_objects.keys():
                    if x <= tracking_objects[key][0][0] <= x + w and y <= tracking_objects[key][0][1] <= y + h:
                        tracking_objects[key].append(h)

        # Setting the starting position of the object of interest to 0
        if first_detect is False and len(tracking_objects.keys()) > index_of_interest:
            start_x = tracking_objects[index_of_interest][0][0]
            print("Start x: " + str(start_x))
            first_detect = True

        if COLLECT_POSITION:
            collect_pos_data(start_x, DATA_STORAGE, tracking_objects, index_of_interest, ALL_INDICES_OF_INTEREST, frame_num)

        # Drawing index values
        for object_id, item in tracking_objects.items():
            cv2.putText(image_with_keypoints, str(object_id), (int(item[0][0] - 5), int(item[0][1] - 17)), 0, 0.5, (0, 255, 0), 1)

        time = round((frame_num - 100) * 0.05, 2)

        # Showing the current frame annotated with the keypoints
        if frame_num >= total_frames - 1:
            cv2.putText(clean_thresh, "Frame: end", (5, 20), 0, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(clean_thresh, "Frame: " + str(frame_num), (5, 20), 0, 0.5, (255, 255, 255), 1)
            cv2.putText(clean_thresh, "Time: " + str(time), (5, 40), 0, 0.5, (255, 255, 255), 1)

        if IMAGE_SAVE:
            save_image('NewShuttleParticleAtTime', time, IMAGE_SAVE_TIMES, clean_thresh)

        # Measurement aid
        cv2.rectangle(image_with_keypoints, TOP_RECT_PT1, TOP_RECT_PT2, RECTANGLE_COLOR, -1)
        cv2.imshow("Frame", image_with_keypoints)
        cv2.waitKey(50)

        frame_num += 1

# ------------------------------------------------------------------------------------------------------------------------


# Releasing the VideoCapture object we created
VID_CAP.release()

# Close any open windows
cv2.destroyAllWindows()
