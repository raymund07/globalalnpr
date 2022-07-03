
import os
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras.preprocessing import image
from  tensorflow.keras.models import load_model




inference_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'inferencegraphs'))

#predictregistration_v1=tf.saved_model.load('{}/character/v1/saved_model'.format(inference_path))
#predictregistration_v2=tf.saved_model.load('{}/character/v2/saved_model'.format(inference_path))
predictregistration=tf.saved_model.load('{}/character/v4/saved_model'.format(inference_path))
predictregistration=tf.saved_model.load('{}/character/mdta/saved_model'.format(inference_path))
predictplate=tf.saved_model.load('{}/plate/v4/saved_model'.format(inference_path))
jurisdiction_model= tf.keras.models.load_model('{}/jurisdiction/mdta'.format(inference_path))





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
    
    def predict_jurisdiction(self,image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        predict=jurisdiction_model.predict(x)
        preds = predict[0]
        data=[]
        labels=['PA', 'DC', 'NH', 'MA', 'DL', 'WV', 'NC', 'GA', 'FL', 'NY', 'MDT', 'VAT', 'VA', 'MD2', 'MD', 'NJ', 'IN', 'MDTS']
        for z in range(0,len(labels)):
            d=labels[z],preds[z]
            data.append(d)
        state=sorted(data,key=lambda x: x[1], reverse=True)[0]
        jurisdiction=state[0]
        jurisdicton_score=state[1]
        return {"state":jurisdiction,'jurisdiction_score':str(jurisdicton_score)}
  



       #test 
        
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
        detections = predictregistration(input_tensor)


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
    a=Inference()
    for images in os.listdir('F:/training/2019-04-19-00-38-04_29-6072-cropped'):

        print(a.predict_jurisdiction('F:/training/2019-04-19-00-38-04_29-6072-cropped/{}'.format(images)))