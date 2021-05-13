
import os
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from utils import label_map_util
from utils import visualization_utils as vis_util
import time
import cv2
import numpy as np

#Add new method if there is a new model for inference

class Inference:
    
    def __init__(self):
        #Initialize and Load model into memory
        # ------------------ Plate Localization------------------------------ 
        self.inference_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..', 'inferencegraphs'))
        self.classes_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..', 'classes'))
        self.base_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..', 'received'))

    
    def detect_plate(self,image_path):
        
        plate_label_map = label_map_util.load_labelmap('{}/plate_localization.pbtxt'.format( self.classes_path))
        plate_categories = label_map_util.convert_label_map_to_categories(plate_label_map, max_num_classes=1, use_display_name=True)
        plate_category_index = label_map_util.create_category_index(plate_categories)
        plate_detection_graph = tf.Graph()
        with plate_detection_graph.as_default():
            plate_od_graph_def = tf.GraphDef()
            with tf.io.gfile.GFile('{}/plate_localization.pb'.format(self.inference_path), 'rb') as f:
                plate_serialized_graph = f.read()
                plate_od_graph_def.ParseFromString(plate_serialized_graph)
                tf.import_graph_def(plate_od_graph_def, name='')

        self.plate_session = tf.Session(graph=plate_detection_graph)
        self.plate_image_tensor = plate_detection_graph.get_tensor_by_name('image_tensor:0')
        self.plate_detection_boxes = plate_detection_graph.get_tensor_by_name('detection_boxes:0')
        self.plate_detection_scores = plate_detection_graph.get_tensor_by_name('detection_scores:0')
        self.plate_detection_classes = plate_detection_graph.get_tensor_by_name('detection_classes:0')
        self.plate_num_detections = plate_detection_graph.get_tensor_by_name('num_detections:0')
        
        start_time = time.time()
        image = cv2.imread(os.path.join(self.base_path,image_path))
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        height = np.size(image, 0)
        width = np.size(image, 1)

        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = self.plate_session.run(
            [self.plate_detection_boxes, self.plate_detection_scores,
                self.plate_detection_classes, self.plate_num_detections],
            feed_dict={self.plate_image_tensor: image_expanded})

        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        boxes = np.squeeze(boxes)
        return (classes,boxes,scores)
        
        
    def detect_char(self,image_path):
        character_label_map = label_map_util.load_labelmap('{}/character_recognition.pbtxt'.format(self.classes_path))
        character_categories = label_map_util.convert_label_map_to_categories(character_label_map, max_num_classes=90, use_display_name=True)
        character_category_index = label_map_util.create_category_index(character_categories)

        character_detection_graph = tf.Graph()

        with character_detection_graph.as_default():
            character_od_graph_def = tf.GraphDef()
            with tf.gfile.GFile('{}/character_recognition.pb'.format(self.inference_path), 'rb') as fid:
                character_serialized_graph = fid.read()
                character_od_graph_def.ParseFromString(character_serialized_graph)
                tf.import_graph_def(character_od_graph_def, name='')
                
        self.character_session = tf.Session(graph=character_detection_graph)
        self.character_image_tensor = character_detection_graph.get_tensor_by_name( 'image_tensor:0')
        self.character_detection_boxes = character_detection_graph.get_tensor_by_name( 'detection_boxes:0')
        self.character_detection_scores = character_detection_graph.get_tensor_by_name('detection_scores:0')
        self.character_detection_classes = character_detection_graph.get_tensor_by_name( 'detection_classes:0')
        self.character_num_detections = character_detection_graph.get_tensor_by_name('num_detections:0')

        start_time = time.time()
        print(os.path.join(self.base_path,'1.jpg'))
        image = cv2.imread('{}'.format(os.path.join(self.base_path,image_path)))
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        height = np.size(image, 0)
        width = np.size(image, 1)

        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = self.character_session.run(
            [self.character_detection_boxes, self.character_detection_scores,
                self.character_detection_classes, self.character_num_detections],
            feed_dict={self.character_image_tensor: image_expanded})

        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        boxes = np.squeeze(boxes)
        return (classes,boxes,scores)
        

    def detect_state(self):
        print('juris')

if __name__ == '__main__':
     plate1=Inference()
     a=plate1.detect_char('1.jpg')
     print(a)
 
