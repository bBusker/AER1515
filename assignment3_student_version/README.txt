dependencies:
see requirements.txt for package dependencies
needs python3.7 or higher
requires same directory structure as in assignment3_student_version code handout
needs assignment2_calibration_code.py (included in zip)

!!!!!needs yolov4 folder in same directory as where scripts are being run from!!!!! 
Overall folder structure should be:
yolov4/
-coco.names
-yolov4.cfg
-yolov4.weights

weights downloaded from https://github.com/AlexeyAB/darknet#pre-trained-models under the "Pre-trained models" header the "yolov4.cfg" and "yolov4.weights" files.

Links also provided here:
yolov4.cfg: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
yolov4.weights: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

They should be put into a directory with the same structure as the provided yolov3 weights (but with the folder name being yolov4) along with the coco.names file
!!!!!needs yolov4 folder in same directory as where scripts are being run from!!!!! 

est_depth contains estimated depth maps for test set
est_segmentation contains estimated segmentation maps for test set



how to run code:
part1:
run part1_estimate_depth.py (default test set)
to run on training set, change flag "train" to True on line 87

part2:
run part2_yolo.py (default test set) (default to using yolov4 weights)
to run on training set, change flag "train" to True on line 161

part3:
run part3_segmentation (default test set) (default to using parameters mentioned in report)
to run on training set, change flag "train" to True on line 165



