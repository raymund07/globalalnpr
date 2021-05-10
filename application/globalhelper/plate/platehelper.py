
import numpy as np

class helper:
 def intersectionOverUnion(self, box1, box2):
    (box1StartY, box1StartX, box1EndY, box1EndX) = box1
    (box2StartY, box2StartX, box2EndY, box2EndX) = box2
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box2StartX, box1StartX)
    yA = max(box2StartY, box1StartY)
    xB = min(box2EndX, box1EndX)
    yB = min(box2EndY, box1EndY)

    # if the boxes are intersecting, then compute the area of intersection rectangle
    if xB > xA and yB > yA:
      interArea = (xB - xA) * (yB - yA)
    else:
      interArea = 0.0

    # compute the area of the box1 and box2
    box1Area = (box1EndY - box1StartY) * (box1EndX - box1StartX)
    box2Area = (box2EndY - box2StartY) * (box2EndX - box2StartX)

    # compute the intersection area / box1 area
    iou = interArea / float(box1Area + box2Area - interArea)

    # return the intersection over area value
    return iou