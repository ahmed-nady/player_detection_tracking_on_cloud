# from sort import *
#
# #create instance of SORT
# mot_tracker = Sort()
#
# # get detections
# ...
#
# # update SORT
# track_bbs_ids = mot_tracker.update(detections)
#
# # track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)

# USAGE
# To read and write back out to video:
# python player_tracking_sort_v2.py --config yolo_v3/yolov3.cfg --weights yolo_v3/yolov3.weights --classes yolo_v3/yolov3.txt --input "F:\PhD\Dataset\Brazil vs Germany Men Football Final Rio 2016.mp4"   #"F:\PhD\Dataset\RELIVE - Ice Hockey - Men's Bronze Medal Game - CANADA vs FINLAND - Day 13   Lausanne 2020.mp4" Ice Hockey - FINLAND - SWITZERLAND - Day 11   Lausanne 2020.mp4" Ice Hockey - SWITZERLAND VS CZECH REPUBLIC.mp4" "F:\PhD\Brazil vs Germany Men Football Final Rio 2016.mp4"
"""
F:\PhD\Dataset\Ice Hockey - SWITZERLAND VS CZECH REPUBLIC.mp4
F:\PhD\Dataset\Brazil vs Germany Men Football Final Rio 2016.mp4
Created on Wed Sep 26 09:28:48 2018

@author: Ahmed Nady
"""
"""
camera7_20101027T210200
camera7_20101027T210300
"""
import numpy as np
import cv2
import argparse
import time
from sort import *

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=True, help="path to yolo config file")
ap.add_argument("-m", "--weights", required=True,
                help="path to yolo pre-trained weight")
ap.add_argument("-i", "--input", type=str,
                help="path to optional input video file")
ap.add_argument("-cl","--classes",required=True,help="path of txt file contains classes")
ap.add_argument("-s", "--skip-frames", type=int, default=10,
	help="# of skip frames between detections")

args = vars(ap.parse_args())

y_start  =320
y_end = 925
def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# initialize the list of class labels  YOLO V3 was trained to
# detect
classes =None
# with open(args["classes"],'r') as f:
#     classes = [line.strip() for line in f.readline()]
with open(args["classes"], 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
#print(classes)

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))


# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold

#create instance of SORT
mot_tracker = Sort()

# initialize the total number of frames processed
 

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNet(args["weights"], args["config"])

if not args.get("input", False):
    print("[INFO] starting video stream")
    vs = cv2.VideoCapture(0)
    time.sleep(2.0)
else:
    print("[INFO] opening video file....")
    vs = cv2.VideoCapture(args["input"])
# #soccer
vs.set(cv2.CAP_PROP_POS_MSEC,504000)
#ice hockey
#vs.set(cv2.CAP_PROP_POS_MSEC,287000)
tracker_count = {}

frm_num =0
while True:
    # get frame from the video
    hasFrame, frame = vs.read()

    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        cv2.waitKey(2000)
        break

    frameCrop = frame.copy()
    detections =[]

    if frm_num % 3==0:
        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        (Height, Width) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (416, 416), (0,0,0),True,crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))
        # loop over the detections

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                # if the class label is not a person, ignore it
                if classes[class_id] != "person":
                    continue
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    # if y+h < y_start or y+h > y_end:
                    #     continue
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            detections.append([round(x),round(y),round(x+w),round(y+h),confidences[i]])
            #print(detections)
            #draw_prediction(frame, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    dets = np.array(detections)
    trackers = mot_tracker.update(dets)

    for d in trackers:
        d = d.astype(np.int32)
        cv2.rectangle(frame, (d[0], d[1]), (d[2] , d[3]), (255,100,100), 2)
        label = 'id = %d' % (d[4])
        cv2.putText(frame, label, (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,100), 2)
        if d[4] in tracker_count:
            tracker_count[d[4]] +=1
        else:
            tracker_count[d[4]] =1

        #save frame every 5 frames
        if (frm_num)%5 ==0:
            #save img crop for player trajectory
            imgCrop = frameCrop[d[1]:d[3],d[0]:d[2]]
            #imgCropResized = cv2.resize(imgCrop,(120,160))
            name = './soccer_tracklets/frm_' + str(d[4]) +'_'+str( tracker_count[d[4]])+ '.png'
            cv2.imwrite(name, imgCrop)
    print("tracker_count",tracker_count)
    if (frm_num)%5 ==0:
        print("********************tracker_count",tracker_count)
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q") :
        break

    # increment the total number of frames processed thus far and
    # then update the FPS counter
    frm_num += 1


vs.release()
# close any open windows
cv2.destroyAllWindows()