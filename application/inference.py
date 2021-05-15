
import os
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tools.utils import label_map_util
import cv2
import numpy as np

inference_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'inferencegraphs'))
plate_detection_graph = tf.Graph()
with plate_detection_graph.as_default():
    plate_od_graph_def = tf.GraphDef()
    with tf.io.gfile.GFile('{}/plate_localization.pb'.format(inference_path), 'rb') as f:
        plate_serialized_graph = f.read()
        plate_od_graph_def.ParseFromString(plate_serialized_graph)
        tf.import_graph_def(plate_od_graph_def, name='')

plate_session = tf.Session(graph=plate_detection_graph)
plate_image_tensor = plate_detection_graph.get_tensor_by_name('image_tensor:0')
plate_detection_boxes = plate_detection_graph.get_tensor_by_name('detection_boxes:0')
plate_detection_scores = plate_detection_graph.get_tensor_by_name('detection_scores:0')
plate_detection_classes = plate_detection_graph.get_tensor_by_name('detection_classes:0')
plate_num_detections = plate_detection_graph.get_tensor_by_name('num_detections:0')

character_detection_graph = tf.Graph()
with character_detection_graph.as_default():
    character_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile('{}/character_recognition.pb'.format(inference_path), 'rb') as fid:
        character_serialized_graph = fid.read()
        character_od_graph_def.ParseFromString(character_serialized_graph)
        tf.import_graph_def(character_od_graph_def, name='')

with character_detection_graph.as_default():
    character_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile('{}/character_recognition.pb'.format(inference_path), 'rb') as fid:
        character_serialized_graph = fid.read()
        character_od_graph_def.ParseFromString(character_serialized_graph)
        tf.import_graph_def(character_od_graph_def, name='')
        
character_session = tf.Session(graph=character_detection_graph)
character_image_tensor = character_detection_graph.get_tensor_by_name( 'image_tensor:0')
character_detection_boxes = character_detection_graph.get_tensor_by_name( 'detection_boxes:0')
character_detection_scores = character_detection_graph.get_tensor_by_name('detection_scores:0')
character_detection_classes = character_detection_graph.get_tensor_by_name( 'detection_classes:0')
character_num_detections = character_detection_graph.get_tensor_by_name('num_detections:0')

#Add new method if there is a new model for inference

class Inference:
    
    def __init__(self):
        #Initialize and Load model into memory
        # ------------------ Plate Localization------------------------------ 
        self.inference_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'inferencegraphs'))
        self.classes_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'classes'))
        self.base_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'received'))

    
    def predict_plate(self,image_path):
        
        plate_label_map = label_map_util.load_labelmap('{}/plate_localization.pbtxt'.format( self.classes_path))
        plate_categories = label_map_util.convert_label_map_to_categories(plate_label_map, max_num_classes=1, use_display_name=True)
        plate_category_index = label_map_util.create_category_index(plate_categories)

        image = cv2.imread(os.path.join(self.base_path,image_path))
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        height = np.size(image, 0)
        width = np.size(image, 1)

        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = plate_session.run(
            [plate_detection_boxes, plate_detection_scores,
                plate_detection_classes, plate_num_detections],
            feed_dict={plate_image_tensor: image_expanded})

        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        boxes = np.squeeze(boxes)
        return (classes,boxes,scores,height,width,image)
        
        
    def predict_registration(self,image_path):
        character_label_map = label_map_util.load_labelmap('{}/character_recognition.pbtxt'.format(self.classes_path))
        character_categories = label_map_util.convert_label_map_to_categories(character_label_map, max_num_classes=90, use_display_name=True)
        character_category_index = label_map_util.create_category_index(character_categories)
  

        image = cv2.imread('{}'.format(os.path.join(self.base_path,image_path)))
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        height = np.size(image, 0)
        width = np.size(image, 1)

        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = character_session.run(
            [character_detection_boxes, character_detection_scores,
                character_detection_classes, character_num_detections],
            feed_dict={character_image_tensor: image_expanded})

        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        boxes = np.squeeze(boxes)
        return (classes,boxes,scores)
        

    def detect_state(self):
        print('juris')

if __name__ == '__main__':
    print('main')