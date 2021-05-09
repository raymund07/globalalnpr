import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
import os
from PIL import Image
import time
import cv2
from globalhelper.utils import label_map_util
from globalhelper.utils import visualization_utils as vis_util

# ------------------ Plate Localization------------------------------ #
inference_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'inferencegraphs'))
classes_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'classes'))
base_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'received'))

# ------------------ Plate Localization------------------------------ #
# ------------------ Character Localization------------------------------ #+
# ------------------ Character Recognition------------------------------ #

# ------------------ Jurisdiction Classification------------------------------ #
# plate_label_map = label_map_util.load_labelmap('{}/plate_localization.pbtxt'.format(classes_path))
# plate_categories = label_map_util.convert_label_map_to_categories(plate_label_map, max_num_classes=1, use_display_name=True)
# plate_category_index = label_map_util.create_category_index(plate_categories)
# plate_detection_graph = tf.Graph()


# with plate_detection_graph.as_default():
#     plate_od_graph_def = tf.GraphDef()
#     with tf.io.gfile.GFile('{}/plate_localization.pb'.format(inference_path), 'rb') as f:
#         plate_serialized_graph = f.read()
#         plate_od_graph_def.ParseFromString(plate_serialized_graph)
#         tf.import_graph_def(plate_od_graph_def, name='')

# plate_session = tf.Session(graph=plate_detection_graph)
# plate_image_tensor = plate_detection_graph.get_tensor_by_name('image_tensor:0')
# plate_detection_boxes = plate_detection_graph.get_tensor_by_name('detection_boxes:0')
# plate_detection_scores = plate_detection_graph.get_tensor_by_name('detection_scores:0')
# plate_detection_classes = plate_detection_graph.get_tensor_by_name('detection_classes:0')
# plate_num_detections = plate_detection_graph.get_tensor_by_name('num_detections:0')
# ---------------------------------------------------------------------------- #

# # ------------------ character Model Initialization ---------------------------- #
character_label_map = label_map_util.load_labelmap('{}/character_recognition.pbtxt'.format(classes_path))
character_categories = label_map_util.convert_label_map_to_categories(character_label_map, max_num_classes=90, use_display_name=True)
character_category_index = label_map_util.create_category_index(character_categories)

character_detection_graph = tf.Graph()

with character_detection_graph.as_default():
    character_od_graph_def = tf.GraphDef()
    with tf.gfile.GFile('{}/character_recognition.pb'.format(inference_path), 'rb') as fid:
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
# # ---------------------------------------------------------------------------- #

# # ------------------ Jurisdiction Detection/Localization------------------------------ #
# jurisdiction_label_map = label_map_util.load_labelmap('{}/jurisdiction_recognition.pbtxt'.format(classes_path))
# jurisdiction_categories = label_map_util.convert_label_map_to_categories(jurisdiction_label_map, max_num_classes=90, use_display_name=True)
# jurisdiction_category_index = label_map_util.create_category_index(jurisdiction_categories)

# jurisdiction_detection_graph = tf.Graph()

# with jurisdiction_detection_graph.as_default():
#     jurisdiction_od_graph_def = tf.GraphDef()
#     with tf.gfile.GFile('{}/jurisdiction_recognition.pb'.format(inference_path), 'rb') as fid:
#         jurisdiction_serialized_graph = fid.read()
#         jurisdiction_od_graph_def.ParseFromString(jurisdiction_serialized_graph)
#         tf.import_graph_def(jurisdiction_od_graph_def, name='')
        
# jurisdiction_session = tf.Session(graph=jurisdiction_detection_graph)
# print(jurisdiction_session)
# jurisdiction_image_tensor = jurisdiction_detection_graph.get_tensor_by_name( 'image_tensor:0')
# jurisdiction_detection_boxes = jurisdiction_detection_graph.get_tensor_by_name( 'detection_boxes:0')
# jurisdiction_detection_scores = jurisdiction_detection_graph.get_tensor_by_name('detection_scores:0')
# jurisdiction_detection_classes = jurisdiction_detection_graph.get_tensor_by_name( 'detection_classes:0')
# jurisdiction_num_detections = jurisdiction_detection_graph.get_tensor_by_name('num_detections:0')



def plate(image_path):
    print('Plate Detection')
    try:
        image = cv2.imread('{}/{}'.format(base_path,image_path))
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
 
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = plate_session.run(
            [plate_detection_boxes, plate_detection_scores,
                plate_detection_classes, plate_num_detections],
            feed_dict={plate_image_tensor: image_expanded})

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
 
    return {"character":object_name, "accuracy":object_score}


def character(image_path):
    print('Character Recognition')
    start_time = time.time()
    image = cv2.imread('{}/{}'.format(base_path,image_path))
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

    object_name = []
    object_score = []
 
#    for i, plateBox in enumerate(plateBoxes_pred):
#       imageResults.append({ 'plateText': charTexts_pred[i], 'plateBoxLoc': list(plateBox),'charBoxLocs': list([list(x) for x in charBoxes_pred[i]])})
#       numPlates += 1

    chars=[]
    platetext=''
    charbox=[]
    charbox.append("[ymin, xmin, ymax, ymax]")
    platescore=[]

    for i,platebox in enumerate(boxes):
        #check online if confidence is >30%
        if(scores[i]>0.30):
            print(scores[i])
            class_name = character_category_index[classes[i]]['name']
            accuracy=scores[i]
            ymin=round(boxes[i][0]*height)
            xmin=round(boxes[i][1]*width)
            ymax=round(boxes[i][2]*height)
            xmax=round(boxes[i][3]*width)
            char=[class_name,scores[i],ymin,xmin,ymax,xmax]
            chars.append(char)
    
    chars = sorted(chars, key=lambda x: x[3])

    for i in range(0,len(chars)):
        print(chars[i][0])
        platetext='{}{}'.format(platetext,chars[i][0])
        charbox.append(str(chars[i][2:]))
  
        platescore.append(str(chars[i][1]*100)[:5])
    print(platetext)
    print(chars)
    print(list(platescore))
    curTime = time.time()
    processingTime = curTime - start_time






    for c in range(0, len(classes)):
        class_name = character_category_index[classes[c]]['name']
        if scores[c] > .30:   # If confidence level is good enough
            
            object_name.append(class_name)
            object_score.append(str(scores[c] * 100)[:5])
    # except:

    #     print("Error occurred in character detection")
    #     object_name = ['']
    #     object_score = ['']

    return {"processingTime":processingTime,"character":platetext, "accuracy":platescore,"boxes":charbox,"imagename":image_path,'model':'character detection'}
  

def jurisdiction(image_path):
    print('Jurisdiction Recognition')

    try:
        image = cv2.imread('{}/{}'.format(base_path,image_path))
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)


        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = plate_session.run(
            [plate_detection_boxes, plate_detection_scores,
                plate_detection_classes, plate_num_detections],
            feed_dict={plate_image_tensor: image_expanded})

        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        boxes = np.squeeze(boxes)

        for i in enumerate(boxes):
            print(i)
    
      

        object_name = []
        object_score = []

        for c in range(0, len(classes)):
            class_name = character_category_index[classes[c]]['name']
            if scores[c] > .30:   # If confidence level is good enough
                object_name.append(class_name)
                object_score.append(str(scores[c] * 100)[:5])
    except:
        print("Error occurred in Jurisdiction detection")
        object_name = ['']
        object_score = ['']

    return {"character":object_name, "accuracy":object_score}


if __name__ == '__main__':
    print('main')
  

