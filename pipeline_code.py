from ultralytics import YOLO
from ultralytics import YOLOWorld
import cv2
import numpy as np

from PIL import Image
from datetime import time, timedelta, datetime

import mysql.connector

#import mysql.connector



conn = mysql.connector.connect(host="localhost",
                               user="root",
                               password = "Programming023@",
                               database="traffic_records")
cur = conn.cursor()

cur.execute("CREATE TABLE IF NOT EXISTS traffic_data (frame_num INT, tracker_id INT,class VARCHAR(255), time_ time,  "
            "light_status VARCHAR(255), jay_status VARCHAR(255), red_violation VARCHAR(255))")
conn.commit()

if conn.is_connected():
    print("Connection established")


#10 frames per second, after 10 frames, inc second count
# start - 6:00:27
# load the yolov8 model


model = YOLOWorld('yolov8s-world.pt')
#model = YOLO('yolov9c.pt')
filter = [0, 1, 2, 3, 5, 7]
#load video
video_path = './traffic_footage.mp4'
cap = cv2.VideoCapture(video_path)

#ret, frame1 = cap.read() #return a new frame and if able to read a new frame
#resize = cv2.resize(frame1,(500,500))
#r = cv2.selectROI('choose traffic light', resize)
#r2 = cv2.selectROI('choose crosswalk', frame)
#x, y, w, h = r
#x1, y1, w1, h1 = r2

# crop the image using the roi
#new_img = resize[y:y + h, x:x + w]

first_frame = True
ret = True
curr_time = datetime.combine(datetime.today(), datetime.strptime("06:00:29", "%H:%M:%S").time())
    #time(6,0,29))

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)

vehicle_count = 0
person_count = 0


fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Using 'mp4v' codec for MP4 format
out = cv2.VideoWriter('yolov8s-test.mp4', fourcc, 10, frame_size)
#read frames
counter = 0
while ret:



    ret, frame = cap.read()  # return a new frame and if able to read a new frame
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    #once we get to 59 seconds we need to update the minute counter

    if frame_number % 10 == 0:  #this means a second has passed (10 frames)
        #second = curr_time.second
        #second += 1
        #curr_time = curr_time.replace(curr_time.hour, curr_time.minute, second)
        curr_time += timedelta(seconds=1)
    #if frame_number % 600 == 0:
        # minute = curr_time.minute
        # minute += 1
        # curr_time = curr_time.replace(curr_time.hour, minute, curr_time.second)
        #curr_time += timedelta(minutes=1)





    #resized_frame = cv2.resize(frame, (500, 500))

    # if its the first frame, do roi to find the region we want
    if first_frame:
        roi1 = cv2.selectROI('choose traffic light', frame)
        roi2 = cv2.selectROI('choose bottom crosswalk', frame) #the main crosswalk opp to signal (bottom)
        roi3 = cv2.selectROI('choose top crosswalk', frame)  # the main crosswalk opp to signal (top)
        roi4 = cv2.selectROI('choose left crosswalk', frame)  # the opposing side (left)
        roi5 = cv2.selectROI('choose right crosswalk', frame)  # the opposing (left)


    first_frame = False
    x, y, w, h = roi1
    r2x1 = int(roi2[0])
    r2x2 = int(roi2[0] + roi2[2])
    r2y1 = int(roi2[1])
    r2y2 = int(roi2[1] + roi2[3])

    r3x1 = int(roi3[0])
    r3x2 = int(roi3[0] + roi3[2])
    r3y1 = int(roi3[1])
    r3y2 = int(roi3[1] + roi3[3])

    r4x1 = int(roi4[0])
    r4x2 = int(roi4[0] + roi4[2])
    r4y1 = int(roi4[1])
    r4y2 = int(roi4[1] + roi4[3])

    r5x1 = int(roi5[0])
    r5x2 = int(roi5[0] + roi5[2])
    r5y1 = int(roi5[1])
    r5y2 = int(roi5[1] + roi5[3])




    if not ret:
        break

    new_img = frame[y:y + h, x:x + w]

    hsv_image = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)


    #hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #convert frame to hsv

    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    lower_red = np.array([160, 100, 100])
    upper_red = np.array([180, 255, 255])

    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    # create a mask with the image and the boundaries, blue regions in white (get the values that are in range of what we want)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

    mask_y = Image.fromarray(yellow_mask)
    mask_g = Image.fromarray(green_mask)
    mask_r = Image.fromarray(red_mask)

    # get the bounding boxes for where the color is detected
    bboxY = mask_y.getbbox()
    bboxG = mask_g.getbbox()
    bboxR = mask_r.getbbox()

    #median filtering
    is_green = False
    is_red = False
    is_yellow = False

    traffic_status = ""

    if bboxY is not None:
        x1, y1, x2, y2 = bboxY
        org_img = cv2.rectangle(new_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, "Yellow", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        is_yellow = True

    if bboxG is not None:
        x1, y1, x2, y2 = bboxG
        org_img = cv2.rectangle(new_img, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(frame, "Green", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        is_green = True

    if bboxR is not None:
        x1, y1, x2, y2 = bboxR
        org_img = cv2.rectangle(new_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "Red", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        is_red = True
    # Add text to the image
    #
    #cv2.rectangle(frame, (x1,y1), (x1+w1,y1+w1),(255,255,255),5)

    #cv2.imwrite('./traffic_light2.jpg', frame)

    if is_green:
        traffic_status = "Green"
    if is_red:
        traffic_status = "Red"
    if is_yellow and is_red:
        traffic_status = "Yellow"
    if is_yellow:
        traffic_status = "Yellow"

    is_jaywalker = "N/A"
    red_dasher = "N/A"


    #detect objects
    #track objects
    results = model.track(frame, persist=True) #we want to remember the frames throughout, so true

    frame_ = results[0].plot()
    boxes = results[0].boxes.xywh.cpu() #all boxes for the frame?
    class_names = results[0].names
    class_ids = results[0].boxes.cls.cpu()
    trackid = results[0].boxes.id
    if trackid is not None:
        track_ids = trackid.int().cpu().tolist()





    ##track_id = 0 #for all the objects in the frame
    for box,classes, id in zip(boxes,class_ids, track_ids):
        x, y, w, h = box
        class_name = class_names[int(classes)]
        track_id = id
        ##track_id += 1
        #inc count each time we see a new car

        if class_name == 'car' or (class_name == "bus") or (class_name == "truck"):
            vehicle_count += 1
            if (y <= r2y1) and (y >= r3y2):  #greater than top of the bottm crosswalk, less than bottom of the top one
                if traffic_status == "Red":
                    red_dasher = "Red Light Violation"
                    cv2.putText(frame_, red_dasher, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)
            cv2.putText(frame_, ("Vehicle Count" + str(vehicle_count)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)


        if class_name == 'person':
            person_count += 1
            if (y >= r2y1) and (y <= r2y2):       #check the bottom crosswalk (test with normal test)
                if (x >= r2x1) and (x <= r2x2):
                    if is_green:
                        is_jaywalker = "jaywalker"

                    if is_red:
                        is_jaywalker = "not jaywalking"
                    cv2.putText(frame_, is_jaywalker, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)
            if (y >= r3y1) and (y <= r3y2):     #check the top crosswalk (test with top (4))
                if (x >= r3x1) and (x <= r3x2):
                    if is_green:
                        is_jaywalker = "jaywalker"

                    if is_red:
                        is_jaywalker = "not jaywalking"
                    cv2.putText(frame_, is_jaywalker, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2,
                                cv2.LINE_AA)

            if (y >= r4y1) and (y <= r4y2):     #check the left crosswalk, flip the logic (test w Left (2))
                if (x >= r4x1) and (x <= r4x2):
                    if is_red:
                        is_jaywalker = "jaywalker"

                    if is_green:
                        is_jaywalker = "not jaywalking"
                    cv2.putText(frame_, is_jaywalker, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2,
                                cv2.LINE_AA)

            if (y >= r5y1) and (y <= r5y2):     #check the right crosswalk
                if (x >= r5x1) and (x <= r5x2):
                    if is_red:
                        is_jaywalker = "jaywalker"

                    if is_green:
                        is_jaywalker = "not jaywalking"
                    cv2.putText(frame_, is_jaywalker, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2,
                                cv2.LINE_AA)
        insert = "INSERT INTO traffic_data (frame_num,tracker_id,class,time_,light_status, jay_status, red_violation)" "VALUES (%s,%s,%s,%s,%s,%s, %s)"
        cur.execute(insert, (frame_number, track_id,class_name, curr_time, traffic_status, is_jaywalker, red_dasher))
        conn.commit()

    counter += 1
    #plot results

    #out = cv2.videowriter(fps,codec,width,hieght)
    out.write(frame_)

    cv2.imshow('frame2', frame_)



    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
