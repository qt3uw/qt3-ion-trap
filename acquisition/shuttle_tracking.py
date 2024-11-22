import cv2
import numpy as np
import math
from tracking_methods import collect_pos_data, set_up_detector, get_frame, post_processing, save_image, setup_tracking

# --------------------------- Config ---------------------------------------------- #

class TrackingConfig:
    def __init__(self):
        # video settings
        self.VIDEO_PATH = '../ExampleSplit.avi'
        self.START_FRAME_NUM = 100

        # regions of interest
        self.X_START = 0
        self.Y_START = 550
        self.X_END = 1600
        self.Y_END = 700

        # image processing
        self.BIN_THRESH = 25
        self.CLEANING_KERNEL = np.ones((2, 2), np.uint8)
        self.FILLING_KERNEL = np.ones((4, 2), np.uint8)

        # frame erasure rectangles
        self.TOP_RECT = ((0, 0), (0, 0))
        self.LEFT_RECT = ((0, 0), (0, 0))
        self.RIGHT_RECT = ((0, 0), (0, 0))
        self.BOTTOM_RECT = ((0, 0), (0, 0))
        self.RECTANGLE_COLOR = (0, 0, 0)

        # tracking settings
        self.STORE_HEIGHT_DATA = False
        self.CONTOUR_DET = False
        self.COLLECT_POSITION = False
        self.ALL_INDICES_OF_INTEREST = []

        # image capture settings
        self.IMAGE_SAVE = False
        self.IMAGE_SAVE_TIMES = [0, 2, 4, 6]

        # data storage
        self.DATA_STORAGE = open('shuttle_data.txt', 'a')

# --------------------------- Video Processing Functions ---------------------------------------------- #

def initialize_video(config):
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, start_frame = get_frame(cap, config.START_FRAME_NUM)
    
    if ret:
        frame_height, frame_width = start_frame.shape[:2]
        print(f"Frame height: {frame_height}\nFrame width: {frame_width}")
        
        roi = start_frame[config.Y_START:config.Y_END, config.X_START:config.X_END]
        cv2.imshow("Frame", roi)
    
    return cap, total_frames, start_frame

# --------------------------- Setup Functions For Tracking ---------------------------------------------- #

def process_frame(config, frame):
    roi_frame = frame[config.Y_START:config.Y_END, config.X_START:config.X_END]
    gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_frame, config.BIN_THRESH, 255, cv2.THRESH_BINARY)
    
    clean_thresh, closing = post_processing(
        thresh,
        config.CLEANING_KERNEL,
        config.FILLING_KERNEL,
        config.RECTANGLE_COLOR,
        *config.TOP_RECT[0], *config.TOP_RECT[1],
        *config.LEFT_RECT[0], *config.LEFT_RECT[1],
        *config.RIGHT_RECT[0], *config.RIGHT_RECT[1],
        *config.BOTTOM_RECT[0], *config.BOTTOM_RECT[1],
        0, 1, 4
    )
    
    return roi_frame, closing, clean_thresh

def update_tracking(tracking_objects, track_id, keypoints_cur_frame, keypoints_prev_frame, contours=None):
    if len(tracking_objects) <= 2:
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
    if frame_num >= total_frames - 1:
        cv2.putText(image, "Frame: end", (5, 20), 0, 0.5, (0, 255, 0), 1)
    else:
        cv2.putText(image, f"Frame: {frame_num}", (5, 20), 0, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"Time: {time}", (5, 40), 0, 0.5, (255, 255, 255), 1)

def draw_tracking_info(image, tracking_objects):
    for object_id, item in tracking_objects.items():
        cv2.putText(image, str(object_id),
                   (int(item[0][0] - 5), int(item[0][1] - 17)),
                   0, 0.5, (0, 255, 0), 1)

# --------------------------- Main Processing Loop ---------------------------------------------- #

def run_tracking(config, cap, detector, total_frames, start_frame):
    frame_num = config.START_FRAME_NUM
    tracking_objects, track_id, keypoints_prev_frame = setup_tracking()
    
    # Initial tracking variables
    index_of_interest = 1
    first_detect = False
    start_x = 0
    
    run = True
    run_body = True
    
    while run:
        frames_to_play = 0
        
        key = cv2.waitKey()
        if key == 27:  # ESC
            run = False
        elif key == 32:  # Space
            frames_to_play = 20
        else:
            frames_to_play = 1

        for _ in range(frames_to_play):
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
            if config.CONTOUR_DET:
                contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Update tracking
            tracking_objects, track_id = update_tracking(
                tracking_objects, track_id, keypoints_cur_frame, 
                keypoints_prev_frame, contours
            )
            
            # set starting position
            if not first_detect and len(tracking_objects.keys()) > index_of_interest:
                start_x = tracking_objects[index_of_interest][0][0]
                print("Start x:", start_x)
                first_detect = True
            
            # collect position data if enabled
            if config.COLLECT_POSITION:
                collect_pos_data(
                    start_x,
                    config.DATA_STORAGE,
                    tracking_objects,
                    index_of_interest,
                    config.ALL_INDICES_OF_INTEREST,
                    frame_num
                )
            
            # draw information
            draw_tracking_info(image_with_keypoints, tracking_objects)
            time = round((frame_num - 100) * 0.05, 2)
            draw_frame_info(clean_thresh, frame_num, time, total_frames)
            
            # if enabled, save image
            if config.IMAGE_SAVE:
                save_image('NewShuttleParticleAtTime', time, 
                          config.IMAGE_SAVE_TIMES, clean_thresh)
            
            # Display frame
            cv2.rectangle(image_with_keypoints, *config.TOP_RECT, config.RECTANGLE_COLOR, -1)
            cv2.imshow("Frame", image_with_keypoints)
            cv2.waitKey(50)
            
            keypoints_prev_frame = keypoints_cur_frame
            frame_num += 1

def main():
    print("Running program...")
    
    config = TrackingConfig()
    
    cap, total_frames, start_frame = initialize_video(config)
    detector = set_up_detector()
    
    run_tracking(config, cap, detector, total_frames, start_frame)
    
    # cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()