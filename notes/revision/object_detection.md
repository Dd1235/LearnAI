# To Learn: Object Detection

Object Locatlization: output the bounding box, eg bx, by, bw, bh, the midpoint and the height and width

eg, three classes: person, car, dog and background
y = [p, bx, by, bw, bh, c1, c2, c3] # if there is an object, its bounding box, and for each classes

For multiple objects, anchor boxes. 

Look at overfeat paper for sliding window approach using covnet. (must faster using convolution)

R-CNN: we extract about 2000 regioni proposals, and then compute cnn features, and classify the regions. (CNN - extract feature, SVM - classify, ann - regress bounding box)

- IOU: a way to evaluate the bounding boxes.

## IOU

Intersection / Union
- IOU: we compute the IOU between the predicted bounding box and the ground truth bounding box. If the IOU is above a certain threshold (e.g., 0.5), we consider it a positive detection; otherwise, it's a negative detection.

## NMS
Non-Maximum Suppression (NMS) is a technique used to eliminate redundant bounding boxes that overlap significantly with each other. The process involves:
- while bounding box:
    - take out the largest probability box
    - remove all other boxes with IOU> threshold
( do this for each class)

## MAP (Mean Average Precision)

- Get all bounding box predictions on our test set.
- For each class, from the confidence, get its its TP or FP etc, calculate the precisiona dn recall. Plot precision recall graph, take the are under the curve, and average it over all classes.

## YOLO

- split image into grid
- the cell with the mid point is responsible for outputting the bounding box and the class
- each output nad label will be relative to the cell.
- label_cell = [c1,..,c20,pc, x, y, w, h]

