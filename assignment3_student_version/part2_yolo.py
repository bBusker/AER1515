# USAGE
# python part2_yolo.py --image images/baggage_claim.jpg --yolo yolo

# import the necessary packages
import numpy as np
import time
import cv2
import os


def detect_cars(image_path):
    ###########################################################
    # OPTIONS
    ###########################################################
    # image_path = 'data/train/left/000001.png'
    yolo_dir = 'yolov4'  # Use YOLOV4 for better accuracy

    # minimum probability to filter weak detections
    confidence_th = 0.5

    # threshold when applying non-maxima suppression
    # higher = less suppression
    threshold = 0.5
    ###########################################################

    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([yolo_dir, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")

    # derive the paths to the YOLO weights and model configurationY
    weightsPath = os.path.sep.join([yolo_dir, "yolov4.weights"])  # Weights path changed to yolov4 weights
    configPath = os.path.sep.join([yolo_dir, "yolov4.cfg"])  # CFG path changed to yolov4 cfgs

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # load our input image and grab its spatial dimensions
    image = cv2.imread(image_path)
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities

    # !!!! Increase the scale input image resolution from 416x416 to 672x672 !!!! #
    # !!!! This reduces runtime but greatly improves performance !!!! #
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (672, 672),
                                 swapRB=True, crop=False)

    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidence_th:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_th,
                            threshold)

    # store which bounding boxes belong to the cars
    car_boxes = []
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            if LABELS[classIDs[i]] == "car":
                car_boxes.append(boxes[i])  # store which bounding boxes belong to the cars
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)

    # return bounding box visualization image and bounding boxes for cars
    return image, np.array(car_boxes)

def detect_boundingboxes(train=True):
    # Set directories
    if train:
        imgs_dir = "./data/train/left"
        output_dir = "./data/train/est_bb"
        sample_list = ['000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010']
    else:
        imgs_dir = "./data/test/left"
        output_dir = "./data/test/est_bb"
        sample_list = ['000011', '000012', '000013', '000014', '000015']
    dirs = [imgs_dir, output_dir]

    # Make dirs
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Run YOLO on each sample
    for sample in sample_list:
        image, boxes = detect_cars(f"{imgs_dir}/{sample}.png")
        cv2.imwrite(f"{output_dir}/{sample}.png", image)  # Save visualization image
        np.save(f"{output_dir}/{sample}", boxes)  # Save car bounding boxes array


if __name__ == "__main__":
    detect_boundingboxes(train=False)