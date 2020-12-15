dependencies:
see requirements.txt for package dependencies
needs python3.7 or higher
requires same directory structure as in assignment3_student_version code handout
needs yolov4 weights extracted to same directory as where scripts are being run from (weights included in zip)

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



