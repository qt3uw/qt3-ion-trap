import cv2
import numpy as np
import math
from tracking_methods import collect_pos_data, set_up_detector, get_frame, post_processing, save_image, TrackingConfig, ShuttlingConfig

# --------------------------- Config ---------------------------------------------- #
"""
class TrackingConfig:
    def __init__(self):


        self.cleaning_kernel = np.ones((2, 2), np.uint8)
        self.filling_kernel = np.ones((4, 2), np.uint8)

        # tracking settings
        self.store_height_data = False
        self.contour_det = False
        self.collect_position = True
        self.all_indices_of_interest = []

        # image capture settings
        self.image_save = True
        self.image_save_times = [0, 2, 4, 6]

        # data storage
        self.data_storage = open('shuttle_data.txt', 'a')
"""

# --------------------------- Video Processing Functions ---------------------------------------------- #

def initialize_video(config):
    cap = cv2.VideoCapture(config.video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, start_frame = get_frame(cap, config.start_frame)
    
    if ret:
        frame_height, frame_width = start_frame.shape[:2]
        print(f"Frame height: {frame_height}\nFrame width: {frame_width}")
        
        roi = start_frame[config.y_range[0]:config.y_range[1], config.x_range[0]:config.x_range[1]]
        cv2.imshow("Frame", roi)
    
    return cap, total_frames, start_frame

# --------------------------- Setup Functions For Tracking ---------------------------------------------- #

def setup_tracking():
    return {}, 0, []  # tracking_objects, track_id, keypoints_prev_frame

def process_frame(config, frame):
    roi_frame = frame[config.y_range[0]:config.y_range[1], config.x_range[0]:config.x_range[1]]
    gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_frame, config.bin_thresh, 255, cv2.THRESH_BINARY)
    
    clean_thresh, closing = post_processing(
        thresh,
        config.cleaning_kernel,
        config.filling_kernel,
        config.rectangle_color,
        config.top_rect[0], config.top_rect[1],
        config.left_rect[0], config.left_rect[1],
        config.right_rect[0], config.right_rect[1],
        config.bottom_rect[0], config.bottom_rect[1],
        0, 1, 4
    )
    
    return roi_frame, closing, clean_thresh

def update_tracking(tracking_objects, track_id, keypoints_cur_frame, keypoints_prev_frame, frame_num, contours=None):
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
    frame_num = config.start_frame
    tracking_objects, track_id, keypoints_prev_frame = setup_tracking()
    
    # Initial tracking variables
    index_of_interest = 0
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
            if config.contour_det:
                contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Update tracking
            tracking_objects, track_id = update_tracking(
                tracking_objects, track_id, keypoints_cur_frame, 
                keypoints_prev_frame, frame_num, contours
            )
            
            # set starting position
            if not first_detect and len(tracking_objects.keys()) > index_of_interest:
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
            time = round((frame_num - 100) * 0.05, 2)
            draw_frame_info(clean_thresh, frame_num, time, total_frames)
            draw_frame_info(image_with_keypoints, frame_num, time, total_frames)
            
            # if enabled, save image
            if config.image_save:
                save_image('NewShuttleParticleAtTime', time, 
                          config.image_save_times, clean_thresh)
            
            # Display frame
            cv2.rectangle(image_with_keypoints, *config.top_rect, config.rectangle_color, -1)
            cv2.imshow("Frame", image_with_keypoints)
            cv2.waitKey(50)
            
            keypoints_prev_frame = keypoints_cur_frame
            frame_num += 1

def main():
    print("Running program...")
    
    config = ShuttlingConfig(video_file = 'ShuttleBackForth.avi', start_frame = 100, x_range = (0, 1600), y_range = (550, 700), \
                            bin_thresh = 45, data_storage = open('shuttle_data.txt', 'a'))
    
    cap, total_frames, start_frame = initialize_video(config)
    detector = set_up_detector()
    
    run_tracking(config, cap, detector, total_frames, start_frame)
    
    # cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
