
import os
import tensorflow as tf
import cv2
import numpy as np


inference_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'inferencegraphs'))

predictregistration_v1=tf.saved_model.load('{}/character/v1/saved_model'.format(inference_path))
#predictregistration_v2=tf.saved_model.load('{}/character/v2/saved_model'.format(inference_path))
#predictregistration_v3=tf.saved_model.load('{}/character/v3/saved_model'.format(inference_path))
predictplate=tf.saved_model.load('{}/plate/saved_model'.format(inference_path))

#Add new method if there is a new model for inference

class Inference:
    
    def __init__(self):
        #Initialize and Load model into memory
        # ------------------ Plate Localization------------------------------ 
        self.inference_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'inferencegraphs'))
        self.classes_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'classes'))
        self.base_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'received'))

    
    def predict_plate(self,image_path):


        
        # plate_label_map = label_map_util.load_labelmap('{}/plate_localization.pbtxt'.format( self.classes_path))
        # plate_categories = label_map_util.convert_label_map_to_categories(plate_label_map, max_num_classes=1, use_display_name=True)
        # plate_category_index = label_map_util.create_category_index(plate_categories)
        path=os.path.join(self.base_path,image_path)
        image = cv2.imread(r'{}'.format(path))
        height = np.size(image, 0)
        width = np.size(image, 1)

 


        image_np = np.array(image)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = predictplate(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections
  

        # image_expanded = np.expand_dims(image, axis=0)
        # (boxes, scores, classes, num) = plate_session.run(
        #     [plate_detection_boxes, plate_detection_scores,
        #         plate_detection_classes, plate_num_detections],
        #     feed_dict={plate_image_tensor: image_expanded})
        # print(scores)

        scores=detections['detection_scores']
        classes= detections['detection_classes'].astype(np.int64)
        boxes = detections['detection_boxes']
        return (classes,boxes,scores,height,width,image)
        
        
    def predict_registration(self,image_path):
     
        # character_label_map = label_map_util.load_labelmap('{}/character_recognition.pbtxt'.format(self.classes_path))
        # character_categories = label_map_util.convert_label_map_to_categories(character_label_map, max_num_classes=90, use_display_name=True)
        # character_category_index = label_map_util.create_category_index(character_categories)
  

        image = cv2.imread('{}'.format(os.path.join(self.base_path,image_path)))
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        height = np.size(image, 0)
        width = np.size(image, 1)

        image_np = np.array(image)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        # if (version=='v1'):
        #     detections = predictregistration_v1(input_tensor)
        # elif (version=='v2'):
        #     detections = predictregistration_v2(input_tensor)
        # elif (version=='v3'):
        #     detections = predictregistration_v3(input_tensor)
        # else:
        detections = predictregistration_v1(input_tensor)


        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections
      

        
        

        # image_expanded = np.expand_dims(image, axis=0)
        # (boxes, scores, classes, num) = plate_session.run(
        #     [plate_detection_boxes, plate_detection_scores,
        #         plate_detection_classes, plate_num_detections],
        #     feed_dict={plate_image_tensor: image_expanded})
        # print(scores)

        scores=detections['detection_scores']
        classes= detections['detection_classes'].astype(np.int64)
        boxes = detections['detection_boxes']
        return (classes,boxes,scores,height,width)

        # image_expanded = np.expand_dims(image, axis=0)
        # (boxes, scores, classes, num) = character_session.run(
        #     [character_detection_boxes, character_detection_scores,
        #         character_detection_classes, character_num_detections],
        #     feed_dict={character_image_tensor: image_expanded})

        # classes = np.squeeze(classes).astype(np.int32)
        # scores = np.squeeze(scores)
        # boxes = np.squeeze(boxes)
        # return (classes,boxes,scores,height,width)
        

    def detect_state(self):
        print('juris')

if __name__ == '__main__':
    print('main')
