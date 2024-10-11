import os
import cv2
import numpy as np
import math

# ----------------- Usage Instructions ----------------- #

'''
First upload the video file of choice and run it in "binary" viewtype to find the object and center the frame on it
with the xrange and yrange values. The surface of the trap must be set to the bottom of the visible frame for 
accurate height data output. A bottom_bar value is likely necessary to cut out the noise from the lowest n frames
Then, set the "auto" feature to True, and the code will run and output data automatically.
If particle tracking is sparse, set "poortracking" to True and input the particle indices to "index_list". Indices can
be identified by manually running the code with auto = False and observing the green value near the object's centroid
'''

# --------------------------- Parameter Definition ---------------------------------------------- #


videofile = cv2.VideoCapture('../ExampleSplit.avi')   # Specifies the video file
viewtype = "image"   # "binary" for binary image, "image" for annotated video, "frame" for original video, "cleanthresh" for undialated
output = "none"   # "tuple" for tuple output, "none" for none
showquant = "none"   # height, micromotion, or both can be displayed on the image
auto = "False"   # Automatically Runs Code (will not display video)
poortracking = False   # Will ignore index of the particle if varying. ONLY USE IF NO OTHER PARTICLES IN FRAME
fps = 20   # Frame rate of the camera
changeinterval = 5   # How many seconds between data point collection
points_to_image = []    # Saves an image of the frame at the specified data point (3, 20, 31, 33, 35 for paper figure 3(a))
sample_frames = 15   # How many frames the data is averaged over
binthresh = 26   # Binary threshold distinguishing white from black (higher for brighter pixels, lower for darker)
xrange = (800, 1200)   # define the visible x range (images are usually 1616 x 1240)
yrange = (547, 933)   # define the visible y range
bottom_bar = 100   # sets the height of bottom blocker. Used to block unwanted noise within the visible frame
top_bar = 0   # sets the height of top blocker
left_bar = 0   # sets the height of left blocker
right_bar = 0   # # sets the height of right blocker

indexlist = []   # Specifies the multiple indices of desired tracking particle in case of sparse tracking/regular index shifts


#-------------------------------Initializing parameter variations-------------------------------------- #


ticker = 0
val = 0

# Sets which frames the data collection starts and ends at. The offset is set to 40% of the spacing between points to ensure equilibrium
offset = (fps * changeinterval) * 0.4
collection_frames = []
for i in range(100):
    collection_frames.append(int((fps * changeinterval * i) + offset))
end_collection_frames = []
for i in range(100):
    end_collection_frames.append(int((collection_frames[i] + sample_frames)))

# Adjusts for inverse y-axis indexing
imageheight = 1240
x_start, x_end = xrange
y_start, y_end = (imageheight-yrange[1]), (imageheight-yrange[0])

# Sets the binary rectangles (blockers)
ylength = y_end - y_start
xlength = x_end - x_start
top_rect_pt1, top_rect_pt2 = (0, 0), (1616, top_bar)
left_rect_pt1, left_rect_pt2 = (0, 0), (left_bar, 1240)
right_rect_pt1, right_rect_pt2 = (xlength-right_bar, 0), (xlength, 1240)
bottom_rect_pt1, bottom_rect_pt2 = (0, ylength-bottom_bar), (1616, ylength)
frameheight = yrange[1]-yrange[0]
if viewtype == "binary":
    rectangle_color = (255, 255, 255)
else: rectangle_color = (0, 0, 0)

# Establishes useful per-datapoint sets for averaging data
microvec = []
heightvec = []
bothvec = []
datapointnum = 0

print("Running program...")


# --------------------------------------- Setting up the detector ------------------------------------------- #


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

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector()
else:
    detector = cv2.SimpleBlobDetector_create(params)


# ---------------------------------- Setting program parameters --------------------------------------------- #


# Choosing the video to track in
vid_cap = videofile
if not vid_cap.isOpened():
    print("Error: Could not open video file.")
    exit()
ret, start_frame = vid_cap.read()
print(ret)

# Object index of interest (Selecting a particle to collect data on)
index_of_interest = 0

# Defining up morphological transformations kernels (Can be different for some optimizations)
cleaning_kernel = np.ones((2, 2), np.uint8)
filling_kernel = np.ones((2, 2), np.uint8)

# Define a starting frame number
start_frame_num = 1


# ------------------------------------------ Priming the main loop --------------------------------------------------#


# Setting up memory between frames for objects
keypoints_prev_frame = []
tracking_objects = {}
track_id = 0

# Frame counter
frame_num = start_frame_num
total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(total_frames)

# Trial counter
trial_num = 1
first_point = False

# Starting point (For clean graphing)
start_x = 0
first_detect = False

# Function to read a specific frame
def get_frame(cap, got_frame_num):
    cap.set(cv2.CAP_PROP_POS_FRAMES, got_frame_num)
    _, got_frame = cap.read()
    return _, got_frame


# ------------------------------------------------- Main Loop ------------------------------------------- #


ret, start_frame = get_frame(vid_cap, frame_num)

start_frame_dim = start_frame.shape
start_frame_height = start_frame_dim[0]
start_frame_width = start_frame_dim[1]
print(f"Frame height: {start_frame_height}\n"
      f"Frame width: {start_frame_width}")

cv2.imshow("Frame", start_frame[y_start:y_end, x_start:x_end])

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
        index_of_interest = int(input('Enter the index of interest: '))
        frames_to_play = sample_frames

        first_point = True
    else:
        if auto == "False":
            frames_to_play = 1
            showvid = True
            datacollect = True
        if auto == "True":
            print('Beginning Autoscan...')
            frames_to_play = 999999
            showvid = False
            datacollect = False


    # Running through the specified number of frames
    for i in range(frames_to_play):
        ret, frame = get_frame(vid_cap, frame_num)
        if not ret and frame_num < total_frames:
            print("Cannot retrieve frame.")
            run = False
        elif frame_num >= total_frames:
            sit = True
            while sit:
                end_control = cv2.waitKey(5)
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
        # print(f"Height: {height}\nWidth: {width}")

        # Creating the region of interest frame
        roi_frame = frame[y_start:y_end, x_start:x_end]
        roi_height = roi_frame.shape[0]
        roi_width = roi_frame.shape[1]
        # print(f"Height: {roi_height}\nWidth: {roi_width}")

        # Converting to black and white
        gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding (Converting to intensity values of 0 and 255)
        ret, thresh = cv2.threshold(gray_frame, binthresh, 255, cv2.THRESH_BINARY)

        # Post-processing
        clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cleaning_kernel, iterations=1)
        cv2.rectangle(clean_thresh, top_rect_pt1, top_rect_pt2, rectangle_color, -1)  # Top Erasure
        cv2.rectangle(clean_thresh, left_rect_pt1, left_rect_pt2, rectangle_color, -1)  # Left Erasure
        cv2.rectangle(clean_thresh, right_rect_pt1, right_rect_pt2, rectangle_color, -1)  # Right Erasure
        cv2.rectangle(clean_thresh, bottom_rect_pt1, bottom_rect_pt2, rectangle_color, -1)  # Bottom erasure
        closing = cv2.morphologyEx(clean_thresh, cv2.MORPH_CLOSE, filling_kernel, iterations=2)

        # Finding the locations of the particles (Will return the top left point of the detected object)
        keypoints = detector.detect(closing)

        keypoints_cur_frame = []
        for keypoint in keypoints:
            point = keypoint.pt
            keypoints_cur_frame.append(keypoint.pt)

        # If the outimage parameter receives an argument of None or an empty array, it will be a copy of the source image
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
            # Create a copy bc we can't remove dict elements while traversing it
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
                        continue  # Moves on to checking next id once we confirm existence

                # Remove lost IDs
                if not object_exists:
                    tracking_objects.pop(object_id)

        # Add found IDs
        for pt1 in keypoints_cur_frame:
            tracking_objects[track_id] = [pt1]
            track_id += 1
            # Iterate over each contour to find the bounding rectangle and get the height
        for i in contours:
            # Get the bounding rectangle for each contour (x, y indicate the top left point of the object)
            x, y, w, h = cv2.boundingRect(i)
            centroid_x, centroid_y = int(x+w/2), int(y+h/2)
            adj_centroid_y = (y_end - y_start) - centroid_y

            # Check which tracking object is in the bounding box
            for key in tracking_objects.keys():
                if x <= tracking_objects[key][0][0] <= x + w and y <= tracking_objects[key][0][1] <= y + h:
                    tracking_objects[key].append(h)

            # Optionally, draw the bounding rectangle on the original image
            if viewtype == "image" and auto == "True":
                showvid = True
            else:
                pass
            if showvid == True:
                # Displays specified quantities on the image
                cv2.rectangle(image_with_keypoints, (x, y), (x + w, y + h), (0, 255, 0), 1)
                if showquant == "height":
                    cv2.putText(image_with_keypoints, 'y: ' + str(adj_centroid_y), (centroid_x, centroid_y + h + 15), 0, 0.5, (0, 255, 0), 1)
                if showquant == "micro":
                    cv2.putText(image_with_keypoints, 'a: ' + str(h), (centroid_x, centroid_y + h + 15), 0, 0.5, (0, 255, 0), 1)
                if showquant == "both":
                    cv2.putText(image_with_keypoints, 'y:' + str(adj_centroid_y), (centroid_x + 12, centroid_y - 8), 0, 0.34, (0, 255, 0), 1)
                    cv2.putText(image_with_keypoints, 'a:' + str(h), (centroid_x + 12, centroid_y + 8), 0,0.34, (0, 255, 0), 1)
            else:
                pass
            if viewtype == "image" and auto == "True":
                showvid = False
            else:
                pass

        # Storing heights detected in "Space bar" frames
        try:
            x, y = point
            adj_y = (y_end - y_start) - y
        except:
            NameError
        if datacollect == True:
            try:
                x, y = point
                adj_y = (y_end - y_start) - y
            except NameError:
                pass
        if poortracking == True:
            for i in range(1000):
                try:
                    height_of_interest = tracking_objects[i][1]
                    if len(tracking_objects.keys()) > 0 and height_of_interest > 0:
                        if i in indexlist:
                            try:
                                heightvec.append(adj_centroid_y)
                                microvec.append(height_of_interest)
                            except KeyError:
                                pass
                            break
                        else:
                            pass
                except KeyError or IndexError:
                    pass
        if poortracking == False:
            try:
                height_of_interest = tracking_objects[index_of_interest][1]  # height_of_interest = tracking_objects[index_of_interest][1]
                if len(tracking_objects.keys()) > 0:
                    try:
                        heightvec.append(adj_centroid_y)
                        microvec.append(height_of_interest)
                    except KeyError:
                        pass
            except KeyError or IndexError:
                pass

        # Puts frame labels and displays requested viewtype frame
        if showvid == True:
            for object_id, item in tracking_objects.items():
                cv2.putText(image_with_keypoints, str(object_id), (int(item[0][0] - 5), int(item[0][1] - 17)), 0, 0.5,
                            (0, 255, 0), 1)
            # Showing the current frame annotated with the keypoints
            if frame_num >= total_frames - 1:
                cv2.putText(image_with_keypoints, "Frame: end", (5, 20), 0, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(image_with_keypoints, "Frame: " + str(frame_num), (5, 20), 0, 0.5, (0, 255, 0), 1)
            if viewtype == "binary":
                cv2.imshow("Frame", closing) #image_with_keypoints, closing
            if viewtype == "image":
                cv2.imshow("Frame", image_with_keypoints)
            if viewtype == "frame":
                cv2.imshow("Frame", frame)

            cv2.waitKey(50)

        # Begins data collection if the frame number is at one of interest
        else:
            if frame_num in collection_frames:
                datacollect = True
            if frame_num in end_collection_frames:
                ticker = ticker + 1
                print(ticker)

                # Identifies if frame is one to save an image from and takes a zoomed in image of the particle
                if ticker in points_to_image:
                    val = val + 1
                    if viewtype == "binary":
                        x, y = point
                        x, y = int(x), int(y)
                        cropped_image = closing[y - 40:y + 40, x - 50:x + 50]
                        x, y = point
                        x, y = int(x), int(y)
                        if val == 1:
                            cv2.imshow("Image", cropped_image)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
                            cv2.imshow('enlarged_image', cropped_image)
                            cv2.imwrite("Frame" + str(points_to_image[val - 1]) + str(viewtype) + "Stacked.tif",
                                        cropped_image)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                        else:
                            cropped_image = closing[y - 40:y + 40, x - 50:x + 50]
                            cv2.imwrite("Frame" + str(points_to_image[val - 1]) + str(viewtype) + "Stacked.tif",
                                        cropped_image)
                    if viewtype == "image":
                        x, y = point
                        x, y = int(x), int(y)
                        cropped_image = image_with_keypoints[y - 50:y + 40, x - 50:x + 50]
                        if val == 1:
                            cv2.imshow("Image", cropped_image)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                        else:
                            pass
                        cv2.imwrite("Frame" + str(points_to_image[val - 1]) + str(viewtype) + "Stacked.tif",
                                    cropped_image)
                    if viewtype == "frame":
                        framed_image = frame[y_start:y_end, x_start:x_end]
                        x,y = point
                        x, y = int(x), int(y)
                        print(x,y)
                        cropped_image = framed_image[y-50:y+50, x-50:x+70]
                        cv2.imwrite("Frame" + str(points_to_image[val - 1]) + str(viewtype) + "Final.tif", cropped_image)
                    print('Click!')

                datapointnum = datapointnum + 1

                # Stores the data in the "Tuple.txt" file and prompts if Tuple.txt already contains data upon beginning collection
                if output == "tuple":
                    avgheight = round(np.mean(heightvec), 2)
                    avgmicro = round(np.mean(microvec), 2)
                    percentage = (frame_num / total_frames) * 100
                    print(str(round(percentage, 0)) + '% , Average Height = ' + str(avgheight) + ' , Average Micromotion = ' + str(avgmicro) + '\n')
                    if os.stat('Tuple.txt').st_size != 0:
                        if datapointnum == 1:
                            acknowledgement = ""
                            while acknowledgement != "continue":
                                acknowledgement = input('Tuple.txt already contains data. Type "continue" to add to the existing file, otherwise stop. ')
                            print("\ncontinuing...")
                        else:
                            both_file = open('Tuple.txt', 'a')
                            both_file.write('[' + str(avgheight) + ', ' + str(avgmicro) + ']\n')
                    else:
                        if datapointnum == 1:
                            both_file = open('Tuple.txt', 'a')
                            both_file.write('[' + str(avgheight) + ', ' + str(avgmicro) + ']\n')
                        else:
                            both_file = open('Tuple.txt', 'a')
                            both_file.write('[' + str(avgheight) + ', ' + str(avgmicro) + ']\n')
                    heightvec = []
                    microvec = []
                datacollect = False
            else:
                pass
            if frame_num == total_frames-1:
                print('Complete')
                break
        frame_num += 1

    # Initiates data collection if auto is off and we are using the spacebar to manually collect the data points
    if frames_to_play == sample_frames:
        if output == "tuple":
            avgheight = round(np.mean(heightvec), 2)
            avgmicro = round(np.mean(microvec), 2)
            print('Average Height = ' + str(avgheight) + ' , Average Micromotion = ' + str(avgmicro))
            if os.stat('Tuple.txt').st_size == 0:
                if datapointnum == 1:
                    input('Tuple.txt already contains data. Press any button to continue to add to it')
                else:
                    both_file = open('Tuple.txt', 'a')
                    both_file.write('[' + str(avgheight) + ', ' + str(avgmicro) + ']\n')


# --------------------------------------- Final Cleanup ---------------------------------------------- #


# Releasing the VideoCapture object we created
vid_cap.release()

# Close any open windows
cv2.destroyAllWindows()

