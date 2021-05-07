

# import numpy as np
# import tensorflow as tf
# import sys
# from PIL import Image
# import cv2

# from utils import label_map_util
# from utils import visualization_utils as vis_util

# # ------------------ Knife Model Initialization ------------------------------ #
# knife_label_map = label_map_util.load_labelmap('training/labelmap.pbtxt')
# knife_categories = label_map_util.convert_label_map_to_categories(
#     knife_label_map, max_num_classes=1, use_display_name=True)
# knife_category_index = label_map_util.create_category_index(knife_categories)

# knife_detection_graph = tf.Graph()

# with knife_detection_graph.as_default():
#     knife_od_graph_def = tf.GraphDef()
#     with tf.gfile.GFile('inference_graph_3/frozen_inference_graph.pb', 'rb') as fid:
#         knife_serialized_graph = fid.read()
#         knife_od_graph_def.ParseFromString(knife_serialized_graph)
#         tf.import_graph_def(knife_od_graph_def, name='')

#     knife_session = tf.Session(graph=knife_detection_graph)

# knife_image_tensor = knife_detection_graph.get_tensor_by_name('image_tensor:0')
# knife_detection_boxes = knife_detection_graph.get_tensor_by_name(
#     'detection_boxes:0')
# knife_detection_scores = knife_detection_graph.get_tensor_by_name(
#     'detection_scores:0')
# knife_detection_classes = knife_detection_graph.get_tensor_by_name(
#     'detection_classes:0')
# knife_num_detections = knife_detection_graph.get_tensor_by_name(
#     'num_detections:0')
# # ---------------------------------------------------------------------------- #

# # ------------------ General Model Initialization ---------------------------- #
# general_label_map = label_map_util.load_labelmap('data/mscoco_label_map.pbtxt')
# general_categories = label_map_util.convert_label_map_to_categories(
#     general_label_map, max_num_classes=90, use_display_name=True)
# general_category_index = label_map_util.create_category_index(
#     general_categories)

# general_detection_graph = tf.Graph()

# with general_detection_graph.as_default():
#     general_od_graph_def = tf.GraphDef()
#     with tf.gfile.GFile('ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb', 'rb') as fid:
#         general_serialized_graph = fid.read()
#         general_od_graph_def.ParseFromString(general_serialized_graph)
#         tf.import_graph_def(general_od_graph_def, name='')

#     general_session = tf.Session(graph=general_detection_graph)

# general_image_tensor = general_detection_graph.get_tensor_by_name(
#     'image_tensor:0')
# general_detection_boxes = general_detection_graph.get_tensor_by_name(
#     'detection_boxes:0')
# general_detection_scores = general_detection_graph.get_tensor_by_name(
#     'detection_scores:0')
# general_detection_classes = general_detection_graph.get_tensor_by_name(
#     'detection_classes:0')
# general_num_detections = general_detection_graph.get_tensor_by_name(
#     'num_detections:0')
# # ---------------------------------------------------------------------------- #


# def knife(image_path):
#     try:
#         image = cv2.imread(image_path)
#         image_expanded = np.expand_dims(image, axis=0)
#         (boxes, scores, classes, num) = knife_session.run(
#             [knife_detection_boxes, knife_detection_scores,
#                 knife_detection_classes, knife_num_detections],
#             feed_dict={knife_image_tensor: image_expanded})

#         classes = np.squeeze(classes).astype(np.int32)
#         scores = np.squeeze(scores)
#         boxes = np.squeeze(boxes)

#         for c in range(0, len(classes)):
#             class_name = knife_category_index[classes[c]]['name']
#             if class_name == 'knife' and scores[c] > .80:
#                 confidence = scores[c] * 100
#                 break
#             else:
#                 confidence = 0.00
#     except:
#         print("Error occurred in knife detection")
#         confidence = 0.0   # Some error has occurred
#     return confidence


# def general(image_path):
#     try:
#         image = cv2.imread(image_path)
#         image_expanded = np.expand_dims(image, axis=0)
#         (boxes, scores, classes, num) = general_session.run(
#             [general_detection_boxes, general_detection_scores,
#                 general_detection_classes, general_num_detections],
#             feed_dict={general_image_tensor: image_expanded})

#         classes = np.squeeze(classes).astype(np.int32)
#         scores = np.squeeze(scores)
#         boxes = np.squeeze(boxes)

#         object_name = []
#         object_score = []

#         for c in range(0, len(classes)):
#             class_name = general_category_index[classes[c]]['name']
#             if scores[c] > .30:   # If confidence level is good enough
#                 object_name.append(class_name)
#                 object_score.append(str(scores[c] * 100)[:5])
#     except:
#         print("Error occurred in general detection")
#         object_name = ['']
#         object_score = ['']

#     return object_name, object_score


# if __name__ == '__main__':
#     print(' in main')

# import the necessary packages
import argparse
import time
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from imutils import paths
from object_detection.utils import label_map_util
from globalhelper.plates.plateFinder import PlateFinder
from globalhelper.plates.predicter import Predicter
from globalhelper.plates.plateDisplay import PlateDisplay
import statistics
import json
def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')
model = tf.Graph()

  # create a context manager that makes this model the default one for
  # execution
with model.as_default():
  # initialize the graph definition
  graphDef = tf.GraphDef()

    # load the graph from disk
  with tf.io.gfile.GFile('datasets/frozen_inference_graph.pb', "rb") as f:
    serializedGraph = f.read()
    graphDef.ParseFromString(serializedGraph)
    tf.import_graph_def(graphDef, name="")

#load the session into memory for faster inference
sess=tf.Session(graph=model)

labelsArg='classes/classes.pbtxt'
imagePathArg="uploadedImages"
num_classesArg=37
min_confidenceArg=0.10
image_displayArg=False
pred_stagesArg=2
# # def predictImages(labelsArg, imagePathArg, num_classesArg, min_confidenceArg, image_displayArg, pred_stagesArg):
def predictImages():

  labelMap = label_map_util.load_labelmap(labelsArg)
  categories = label_map_util.convert_label_map_to_categories(labelMap, max_num_classes=num_classesArg,use_display_name=True)
  categoryIdx = label_map_util.create_category_index(categories)

  plateFinder = PlateFinder(min_confidenceArg, categoryIdx,
                            rejectPlates=False, charIOUMax=0.3)
  predicter = Predicter(model, sess, categoryIdx)

  imagePaths = paths.list_images(imagePathArg)
  frameCnt = 0
  start_time = time.time()
  platesReturn = []
  numPlates = 0
  # Loop over all the images
  for imagePath in imagePaths:
    frameCnt += 1

    # load the image from disk

    image = cv2.imread(imagePath)
    (H, W) = image.shape[:2]

    # If prediction stages == 2, then perform prediction on full image, find the plates, crop the plates from the image,
    # and then perform prediction on the plate images
    if pred_stagesArg == 2:
      # Perform inference on the full image, and then select only the plate boxes
      boxes, scores, labels = predicter.predictPlates(image, preprocess=True)
      licensePlateFound_pred, plateBoxes_pred, plateScores_pred = plateFinder.findPlatesOnly(boxes, scores, labels)
      # loop over the plate boxes, find the chars inside the plate boxes,
      # and then scrub the chars with 'processPlates', resulting in a list of final plateBoxes, char texts, char boxes, char scores and complete plate scores
      plates = []

      for plateBox in plateBoxes_pred:
        boxes, scores, labels = predicter.predictChars(image, plateBox)
        chars = plateFinder.findCharsOnly(boxes, scores, labels, plateBox, image.shape[0], image.shape[1])
        if len(chars) > 0:
          plates.append(chars)
        else:
          plates.append(None)
      plateBoxes_pred, charTexts_pred, charBoxes_pred, charScores_pred, plateAverageScores_pred = plateFinder.processPlates(plates, plateBoxes_pred, plateScores_pred)
      print(charScores_pred)
    # If prediction stages == 1, then predict the plates and characters in one pass
    elif pred_stagesArg == 1:
      # Perform inference on the full image, and then find the plate text associated with each plate
      boxes, scores, labels = predicter.predictPlates(image, preprocess=False)
      licensePlateFound_pred, plateBoxes_pred, charTexts_pred, charBoxes_pred, charScores_pred, plateScores_pred = plateFinder.findPlates(
        boxes, scores, labels)
   
   

    else:
      print("[ERROR] --pred_stages {}. The number of prediction stages must be either 1 or 2".format(pred_stagesArg))
      quit()

    # Print plate text
  
  
    


    # for charText in charTexts_pred:
    #   print("    Found: ", charText)

    # Display the full image with predicted plates and chars
    if image_displayArg == True:
      imageLabelled = plateDisplay.labelImage(image, plateBoxes_pred, charBoxes_pred, charTexts_pred,charScores_pred)
      cv2.imshow("Labelled Image", imageLabelled)
      cv2.waitKey(0)


    print(charBoxes_pred)
    print(plateBox)

 
    imageResults = []
    for i, plateBox in enumerate(plateBoxes_pred):
      imageResults.append({ 'plateText': charTexts_pred[i], 'plateBoxLoc': list(plateBox),'charBoxLocs': list([list(x) for x in charBoxes_pred[i]])})
      numPlates += 1

    platesReturn.append({'imagePath': imagePath.split("/")[-1], 'imageResults': imageResults})

  # print some performance statistics
  curTime = time.time()
  processingTime = curTime - start_time
  fps = frameCnt / processingTime
  print("[INFO] Processed {} frames in {:.2f} seconds. Frame rate: {:.2f} Hz".format(frameCnt, processingTime, fps))

 
  return {"processingTime": processingTime,  "numPlates": numPlates, "numImages": len(platesReturn), "images": platesReturn}



if __name__ == '__main__':
  #tf.app.run()
  # construct the argument parse and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-m", "--model",default='datasets/experiment_ssd/1/exported_model/frozen_inference_graph.pb ',required=False,
                  help="base path for frozen checkpoint detection graph")
  ap.add_argument("-l", "--labels", default='classes/classes.pbtxt', required=False,
                  help="labels file")
  ap.add_argument("-i", "--imagePath",default='images', required=False,
                  help="path to input image path")
  ap.add_argument("-n", "--num-classes",default=37, type=int, required=False,
                  help="# of class labels")
  ap.add_argument("-c", "--min-confidence", type=float, default=0.10,
                  help="minimum probability used to filter weak detections")
  ap.add_argument("-d", "--image_display", type=str2bool, default=True,
                  help="Enable display of annotated images")
  ap.add_argument("-p", "--pred_stages", default=2,type=int, required=False,
                  help="number of prediction stages")

  args = vars(ap.parse_args())
  results = predictImages(args["model"], args["labels"], args["imagePath"], args["num_classes"], args["min_confidence"],
                 args["image_display"], args["pred_stages"])
