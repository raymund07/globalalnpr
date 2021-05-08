import numpy as np
import tensorflow as tf
import sys
from PIL import Image
import cv2
from utils import label_map_util
from utils import visualization_utils as vis_util

# ------------------ Plate Localization------------------------------ #
# ------------------ Character Localization------------------------------ #+
# ------------------ Character Recognition------------------------------ #
# ------------------ Jurisdiction Detection/Localization------------------------------ #
# ------------------ Jurisdiction Classification------------------------------ #
plate_location = label_map_util.load_labelmap('plate/labelmap.pbtxt')
plate_categories = label_map_util.convert_label_map_to_categories(plate_label_map, max_num_classes=1, use_display_name=True)
plate_category_index = label_map_util.create_category_index(plate_categories)

plate_detection_graph = tf.Graph()

with plate_detection_graph.as_default():
    plate_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile('Inferencegraphs/plate_localization.pb', 'rb') as fid:
        plate_serialized_graph = fid.read()
        plate_od_graph_def.ParseFromString(plate_serialized_graph)
        tf.import_graph_def(plate_od_graph_def, name='')

plate_session = tf.Session(graph=plate_detection_graph)

plate_image_tensor = plate_detection_graph.get_tensor_by_name('image_tensor:0')
plate_detection_boxes = plate_detection_graph.get_tensor_by_name('detection_boxes:0')
plate_detection_scores = plate_detection_graph.get_tensor_by_name('detection_scores:0')
plate_detection_classes = plate_detection_graph.get_tensor_by_name('detection_classes:0')
plate_num_detections = plate_detection_graph.get_tensor_by_name('num_detections:0')
# ---------------------------------------------------------------------------- #

# ------------------ character Model Initialization ---------------------------- #
character_label_map = label_map_util.load_labelmap('classes/character_recognition.pbtxt')
character_categories = label_map_util.convert_label_map_to_categories(character_label_map, max_num_classes=90, use_display_name=True)
character_category_index = label_map_util.create_category_index(character_categories)

character_detection_graph = tf.Graph()

with character_detection_graph.as_default():
    character_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile('Inferencegraphs/character_recognition.pb', 'rb') as fid:
        character_serialized_graph = fid.read()
        character_od_graph_def.ParseFromString(character_serialized_graph)
        tf.import_graph_def(character_od_graph_def, name='')
character_session = tf.Session(graph=character_detection_graph)
print(character_session)
character_image_tensor = character_detection_graph.get_tensor_by_name( 'image_tensor:0')
character_detection_boxes = character_detection_graph.get_tensor_by_name( 'detection_boxes:0')
character_detection_scores = character_detection_graph.get_tensor_by_name('detection_scores:0')
character_detection_classes = character_detection_graph.get_tensor_by_name( 'detection_classes:0')
character_num_detections = character_detection_graph.get_tensor_by_name('num_detections:0')
# ---------------------------------------------------------------------------- #


def plate(image_path):
    try:
        image = cv2.imread(image_path)
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = plate_session.run(
            [plate_detection_boxes, plate_detection_scores,
                plate_detection_classes, plate_num_detections],
            feed_dict={plate_image_tensor: image_expanded})

        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        boxes = np.squeeze(boxes)

        for c in range(0, len(classes)):
            class_name = plate_category_index[classes[c]]['name']
            if class_name == 'plate' and scores[c] > .80:
                confidence = scores[c] * 100
                break
            else:
                confidence = 0.00
    except:
        print("Error occurred in plate detection")
        confidence = 0.0   # Some error has occurred
    return confidence


def character(image_path):
    try:
        image = cv2.imread(image_path)
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = character_session.run(
            [character_detection_boxes, character_detection_scores,
                character_detection_classes, character_num_detections],
            feed_dict={character_image_tensor: image_expanded})

        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        boxes = np.squeeze(boxes)

        object_name = []
        object_score = []

        for c in range(0, len(classes)):
            class_name = character_category_index[classes[c]]['name']
            if scores[c] > .30:   # If confidence level is good enough
                object_name.append(class_name)
                object_score.append(str(scores[c] * 100)[:5])
    except:
        print("Error occurred in character detection")
        object_name = ['']
        object_score = ['']

    return object_name, object_score


if __name__ == '__main__':
    print(' in main')

