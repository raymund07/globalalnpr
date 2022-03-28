import os
import tensorflow as tf
from object_detection.utils import label_map_util
import tools.utils.visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import csv
import time
import cv2 
import numpy as np
from PIL import Image
import os
import requests
import cv2
import json
import numpy as np
from time import sleep
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras.preprocessing import image
from  tensorflow.keras.models import load_model
from tools.detector import Detector
from flask import Flask, render_template, request, jsonify
from numpy.testing._private.utils import print_assert_equal
from werkzeug.utils import secure_filename
import base64
import io
from PIL import Image

# from inference import Inference
import shutil
import os
from flask import Flask, Response
from gevent.pywsgi import WSGIServer
from gevent import monkey
# inference_path='C:/Users/Isaac/global/tensorflow-anpr/inferencegraphs'
# # jurisdiction_model = load_model('{}/jurisdiction/mdta'.format(inference_path))
# jurisdiction_model= tf.keras.models.load_model('{}/jurisdiction/mdta'.format(inference_path))

from mss import mss
import cv2
from PIL import Image
import numpy as np
from time import time
import os
import requests
import json
import cv2
import threading
import argparse
import csv
import pprint
import pyautogui as p
import keyboard as k

from time import sleep
mon = {'top': 282, 'left':288, 'width':(628-288), 'height':(427-282)}
# mon = {'top': 196, 'left':140, 'width':(750-140), 'height':(427-196)}
mon = {'top': 300, 'left':292, 'width':(1306-292), 'height':(701-300)}
# mon = {'top': 204, 'left':149, 'width':(724-149), 'height':(506-204)}
# mon2 = {'top': 204, 'left':1124, 'width':(1654-1124), 'height':(506-204)}
plate=''
keys=[]

from pprint import pprint

from PIL import Image, ImageDraw, ImageFont
application = Flask(__name__)
sct = mss()
img1=''
mp=p.position()


# category_index = label_map_util.create_category_index_from_labelmap('annotations/character/label_map.pbtxt')
# configs = config_util.get_configs_from_pipeline_file('models/efficientnetB0/charlocation.config')
# detection_model = model_builder.build(model_config=configs['model'], is_training=False)
category_index = label_map_util.create_category_index_from_labelmap('annotations/label_map.pbtxt')
configs = config_util.get_configs_from_pipeline_file('annotations/character.config')
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# conn = mysql.connector.connect(
#    user='root', password='', host='127.0.0.1', database='alnpr')
# c=conn.cursor()
states="TX"

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)

#ckpt.restore('models/efficientnetB0/character/ckpt-49').expect_partial()#TEMP PLATE character-detection\FL
# ckpt.restore('models/efficientnetB0/character-detection/TXS/ckpt-56').expect_partial()#TEMP PLATE character-detection\FL
# ckpt.restore('models/efficientnetB0/character-detection/TXDV/ckpt-56').expect_partial()#TXDV
# ckpt.restore('models/efficientnetB0/character-detection/TX/ckpt-24').expect_partial()#TX
# ckpt.restore('models/efficientnetB0/character-detection/TXS/ckpt-56').expect_partial()#TX
# ckpt.restore('models/efficientnetB0/character-detection/CA/ckpt-6').expect_partial()#TX
# ckpt.restore('models/efficientnetB0/character-recog/v3/ckpt-39').expect_partial()#TX
ckpt.restore('annotations/ckpt-12').expect_partial()#TX

# predict_character=tf.saved_model.load('exported-models/character/saved_model')
def select(a):
    
    c.execute("SELECT plate FROM images where image_name=%s",(a,))
    #c.execute("SELECT i.ImageID,i.Path,i.folder_date,i.folder_datetime,i.Plate,i.CompositeScore,i.JCode,0401_alpr.ID FROM images i left join 0401_alpr on i.ImageId=0401_alpr.ImageID where i.folder_date='2019-04-01' and i.JCode!='TX ' and 0401_alpr.ID IS NULL")
    result=c.fetchone()
    conn.commit()
    return result
def select_data():
    
    c.execute("SELECT plate,path FROM 0401_alpr where jurisdiction='TX' and jurisdiction_score>.90")
    #c.execute("SELECT i.ImageID,i.Path,i.folder_date,i.folder_datetime,i.Plate,i.CompositeScore,i.JCode,0401_alpr.ID FROM images i left join 0401_alpr on i.ImageId=0401_alpr.ImageID where i.folder_date='2019-04-01' and i.JCode!='TX ' and 0401_alpr.ID IS NULL")
    result=c.fetchall()
    conn.commit()
    return result


def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def write_csv(image_name,width,height,classes,xmin,ymin,xmax,ymax):
   
     file_exists = os.path.isfile('C:/Users/Isaac/Desktop/mdta/mdta.csv')
     header = ['filename', 'width', 'height', 'class','xmin', 'ymin', 'xmax', 'ymax']
     with open('C:/Users/Isaac/Desktop/mdta/mdta.csv', 'a', encoding='UTF8', newline='') as f:
          fieldname= ['filename', 'width', 'height', 'class','xmin', 'ymin', 'xmax', 'ymax']
          writer = csv.DictWriter(f,fieldnames=fieldname)
          if not file_exists:
               writer.writeheader()  # file doesn't exist yet, write a header
          writer.writerow({'filename':image_name, 'width':width, 'height':height, 'class':classes,'xmin':xmin, 'ymin':ymin, 'xmax':xmax, 'ymax':ymax})
plate=''
keys=[]
registration=''
base_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'received'))
def predict_jurisdiction(img):
    # img = image.load_img(image_path, target_size=(299, 299))
    # x = image.img_to_array(img)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    predict=jurisdiction_model.predict(x)
    preds = predict[0]
    data=[]
    # labels=['AL', 'AR', 'AZ', 'CA', 'CO', 'FL', 'GA', 'IL', 'IN', 'LA1', 'LA2', 'MO', 'NJ', 'NY', 'OH', 'OK', 'PA', 'TN', 'TX', 'TXDV', 'TXS', 'TXT', 'VA']
    labels=['MD','MD2', 'MDTS', 'MDT','VA']
    for z in range(0,len(labels)):
        d=labels[z],preds[z]
        data.append(d)
    state=sorted(data,key=lambda x: x[1], reverse=True)[0]
    jurisdiction=state[0]
    jurisdicton_score=state[1]
    return {"state":jurisdiction,'jurisdiction_score':str(jurisdicton_score)}


def on_press(key):
    k=str(key).replace("'","")
    
    if k=='a' or k=='b' or k=='c' or k=='d' or k=='e' or k=='f' or k=='g' or k=='h' or k=='i' or k=='j' or k=='k' or k=='l' or k=='m' or k=='n' or  k=='o' or k=='p' or k=='q' or k=='r' or k=='s' or k=='t' or k=='u' or k=='v' or k=='w' or k=='x' or k=='y' or k=='z' or k=='1' or k=='2' or k=='3' or k=='4' or k=='5' or k=='6' or k=='7' or k=='8' or k=='9' or k=='0': 
        keys.append(k)
    if k=='Key.backspace' and len(keys)>=1:
        keys.pop()
    print(keys)
def on_release(key):
    if key==Key.enter:
        keys=[]
        return False
@application.route('/api/v2' , methods=['POST'])     
def predict(sct_img):
    file = request.files['image']
    begin_time = time()
    print(begin_time);
    filename = secure_filename(file.filename)
    file.save('{}/{}'.format(base_path,filename))
    img_bgr=cv2.imread('{}/{}'.format(base_path,filename))
    # img = Image.frombytes('RGB', (sct_img.size.width, sct_img.size.height), sct_img.rgb)
    # img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # img_bgr = cv2.resize(img_bgr, (488,231), interpolation = cv2.INTER_AREA)
    # img_jurs=cv2.resize(img_bgr, (299,299), interpolation = cv2.INTER_AREA)
    image_np = np.array(img_bgr)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    #input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    scores=detections['detection_scores']
    classes= detections['detection_classes'].astype(np.int64)
    platescore=[]
    chars=[]
    platetext=''
    height = np.size(img_bgr, 0)
    width = np.size(img_bgr, 1)
    for j,platebox in enumerate(detections['detection_boxes']):
        if(scores[j]>=.80):
            class_name = category_index[classes[j]+1]['name']
            xmin=int(platebox[1]*width)
            ymin=int(platebox[0]*height)
            ymax=int(platebox[2]*height)
            xmax=int(platebox[3]*width)
            char=[class_name,round(scores[j],2),xmin,ymin,xmax,ymax]
            chars.append(char)
            label_id_offset = 1
            image_np_with_detections = image_np.copy()
            # write_csv(name,width,height,class_name,xmin,ymin,xmax,ymax)
            a=viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],

                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=20,
                min_score_thresh=.80,
                agnostic_mode=False) 
    chars = sorted(chars, key=lambda x: x[2])
    print(time()-begin_time)
    stack=[]
    original=[]
    pl_regis=[]
    for l in range(0,len(chars)):
        char_height=chars[l][5]-chars[l][3]

        if char_height<=50:
            original.append(l)

            stack.append(chars[l])
    sorted_license = sorted(stack, key=lambda x: x[3]) 
    if(sorted_license==stack):
        print('correct')
    if(sorted_license!=stack and len(sorted_license)==2):

        print(sorted_license)

        chars[original[0]]=sorted_license[0]
        chars[original[1]]=sorted_license[1]

    result={}
    for i in range(0,len(chars)):
        platetext='{}{}'.format(platetext,chars[i][0])
        platescore.append(str(chars[i][1]*100)[:5])
        result={'licenseplate':platetext , "score":platescore}
    
    # juris=predict_jurisdiction(img_jurs)
    # print(juris)
    return platetext



if __name__ == '__main__':
  # application.run(host='0.0.0.0',port=5000,debug = True)
  #  application.run(port=5000,debug = True)
   http = WSGIServer(('0.0.0.0', 5000), application.wsgi_app) 

    # Serve your application
  http.serve_forever()



# if __name__ == "__main__":
    

#     while 1:
#         registration=[]
#         imagelist=[]
#         # imagelist.append(sct.grab(mon))
#         # # imagelist.append(sct.grab(mon2))
#         # for image in imagelist:
#         #     result=predict(image)  
#         #     registration.append(result)
#         # print(registration)
#         # p.click(288,282)
#         # p.press('space')
#         # k.write(registration[0])
#         # p.press('tab')
#         # p.press('1')
#         # p.press('space')
        
#         if k.is_pressed('+'):
#             imagelist.append(sct.grab(mon))
#             # imagelist.append(sct.grab(mon2))
#             for image in imagelist:
#                 result=predict(image)
#                 registration.append(result)
#             print(registration)
#             p.click(288,282)
#             p.press('space')
#             p.typewrite(registration[0])
#             p.press(['tab','1','space'])
     
#             # p.click(1859,641)
#             # p.press('space')
#             # k.write(registration[1])
#             # p.press('tab')
#             # p.press('1')
#             # p.press('space')
#             # sleep(1)
                
       
            

#             # print(chars)
  
#             # if len(platetext)==len(registration):
#             #     print('same')
#             #     for i in range(0,len(registration)):
#             #         xmin=chars[i][2]
#             #         ymin=chars[i][3]
#             #         xmax=chars[i][4]
#             #         ymax=chars[i][5]
#             #         print(xmin,ymin,xmax,ymax)
#             #         write_csv("{}.JPG".format(begin_time),width,height,registration[i],xmin,ymin,xmax,ymax)
#             #     sleep(.2)
#             #     k.write(registration)
#             #     k.press('tab')
#             #     sleep(.1)
#             #     k.press('1')
#             #     sleep(.1)
#             # if len(platetext)!=len(registration):
#             #     cv2.imwrite('C:/Users/Isaac/Desktop/mis/{}.JPG'.format(begin_time),img_bgr)
  
    












#     # plate_count=[]


#     # base='C:/Users/Isaac/Desktop/images'
#     # for i in os.listdir('C:/Users/Isaac/Desktop/images'):
#     #     print(i)
#     #     if i.endswith('.xml'):
#     #         continue
#     #     else:
#     #         if os.path.exists('{}/{}'.format(base,i)):
#     #             print('{}/{}'.format(base,i))
#     #             img = cv2.imread('{}/{}'.format(base,i))
#     #             height = np.size(img, 0)
#     #             width = np.size(img, 1)
#     #             print(height,width)
#     #             image_np = np.array(img)
#     #             input_tensor = tf.convert_to_tensor(image_np)

#     #             input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
#     #             #input_tensor = input_tensor[tf.newaxis, ...]
#     #             start_time = time.time()
#     #             detections = detect_fn(input_tensor)


                

#     #             num_detections = int(detections.pop('num_detections'))
#     #             detections = {key: value[0, :num_detections].numpy()
#     #                         for key, value in detections.items()}
#     #             detections['num_detections'] = num_detections
#     #             print(time.time()-start_time)

#     #             # detection_classes should be ints.
#     #             detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

#     #             scores=detections['detection_scores']
#     #             classes= detections['detection_classes'].astype(np.int64)
#     #             platescore=[]
#     #             chars=[]
#     #             platetext=''
#     #             print(classes)     
#     #             for j,platebox in enumerate(detections['detection_boxes']):
#     #                 if(scores[j]>=.90):
#     #                     class_name = category_index[classes[j]+1]['name']
#     #                     xmin=int(platebox[1]*width)
#     #                     ymin=int(platebox[0]*height)
#     #                     ymax=int(platebox[2]*height)
#     #                     xmax=int(platebox[3]*width)
#     #                     char=[class_name,round(scores[j],2),xmin,ymin,xmax,ymax]
#     #                     chars.append(char)
#     #                     label_id_offset = 1
#     #                     image_np_with_detections = image_np.copy()
#     #                     # write_csv(name,width,height,class_name,xmin,ymin,xmax,ymax)


#     #                     a=viz_utils.visualize_boxes_and_labels_on_image_array(
#     #                         image_np_with_detections,
#     #                         detections['detection_boxes'],
#     #                         detections['detection_classes']+label_id_offset,
#     #                         detections['detection_scores'],
#     #                         category_index,
#     #                         use_normalized_coordinates=True,
#     #                         max_boxes_to_draw=8,
#     #                         min_score_thresh=.90,
#     #                         agnostic_mode=False) 
                       
#     #             cv2.imwrite('{}/test/{}.JPG'.format(base,i),a)


                          


          

#     # for plate in data:

      
#         # image_path=('{}{}'.format(base,plate[1])).replace('\\','/')
#         # name=image_path.split('/')[-1]
       
#         # if i.endswith('.xml'):
#         #     continue
#         # if os.path.exists(image_path):
           
#         #     # q=select(i)
            
#         #     # print(q)

#         #     # img = cv2.imread('F:/training/character/{}/{}'.format(states,i))
#         #     img = cv2.imread(image_path)
        

#         #     height = np.size(img, 0)
#         #     width = np.size(img, 1)



#         #     image_np = np.array(img)
#         #     input_tensor = tf.convert_to_tensor(image_np)

#         #     input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
#         #     #input_tensor = input_tensor[tf.newaxis, ...]
#         #     start_time = time.time()
#         #     detections = detect_fn(input_tensor)


            

#         #     num_detections = int(detections.pop('num_detections'))
#         #     detections = {key: value[0, :num_detections].numpy()
#         #                 for key, value in detections.items()}
#         #     detections['num_detections'] = num_detections
#         #     print(time.time()-start_time)

#         #     # detection_classes should be ints.
#         #     detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

#         #     scores=detections['detection_scores']
#         #     classes= detections['detection_classes'].astype(np.int64)
#         #     platescore=[]
#         #     chars=[]
#         #     platetext=''
            
        



#         #     for i,platebox in enumerate(detections['detection_boxes']):
#         #         if(scores[i]>=.90):
#         #             class_name = category_index[classes[i]+1]['name']
#         #             xmin=int(platebox[1]*width)
#         #             ymin=int(platebox[0]*height)
#         #             ymax=int(platebox[2]*height)
#         #             xmax=int(platebox[3]*width)
#         #             char=[class_name,round(scores[i],2),xmin,ymin,xmax,ymax]
#         #             chars.append(char)
#         #             label_id_offset = 1
#         #             image_np_with_detections = image_np.copy()
#         #             # write_csv(name,width,height,class_name,xmin,ymin,xmax,ymax)


#         #             a=viz_utils.visualize_boxes_and_labels_on_image_array(
#         #                 image_np_with_detections,
#         #                 detections['detection_boxes'],
#         #                 detections['detection_classes']+label_id_offset,
#         #                 detections['detection_scores'],
#         #                 category_index,
#         #                 use_normalized_coordinates=True,
#         #                 max_boxes_to_draw=8,
#         #                 min_score_thresh=.90,
#         #                 agnostic_mode=False)
            
#         #     if not os.path.exists('F:/training/character/DRAW/{}'.format(states)):
#         #         os.mkdir('F:/training/character/DRAW/{}'.format(states))
                
#         #     cv2.imwrite('F:/training/character/DRAW/{}/{}'.format(states,name),a)
#         #     chars = sorted(chars, key=lambda x: x[2])
#         #     result={}
#         #     for i in range(0,len(chars)):
#         #         platetext='{}{}'.format(platetext,chars[i][0])
#         #         platescore.append(str(chars[i][1]*100)[:5])
#         #     result={'licenseplate':platetext , "score":platescore}
#         #     if (len(platetext)==len(plate[0])):
#         #         for i in range(0,len(chars)):
#         #             write_csv(name,width,height,plate[0][i],xmin,ymin,xmax,ymax)


            

            
#     #         # # crop_img = img[ymin:ymax, xmin:xmax]
                    

#     #         # # cv2.imwrite('images/cropped/{}'.format(name),crop_img)

#     #         chars = sorted(chars, key=lambda x: x[2])
#     #         result={}
#     #         for i in range(0,len(chars)):
            
#     #             platetext='{}{}'.format(platetext,chars[i][0])

#     #             platescore.append(str(chars[i][1]*100)[:5])
#     #             result={'licenseplate':platetext , "score":platescore}

#     #         if (len(platetext)==len(q[0])):
#     #             print('same')
#     #             for i in range(0,len(q[0])):
                    
                    
#     #                 xmin=chars[i][2]
#     #                 ymin=chars[i][3]
#     #                 xmax=chars[i][4]
#     #                 ymax=chars[i][5]
                    

                    
#     #                 # write_csv(name,width,height,q[0][i],xmin,ymin,xmax,ymax)
#     #     except:
#     #         print(q)
#     #         print(name)

#         # if (len(platetext)==7 and a!=''):
        
#         #     cv2.imwrite('{}/character-recognition-test/art1/{}'.format(base_path,name),a)
#         #     a=''
            
#         # elif(len(platetext)<7 and a!='' ):
        
#         #     print('length is less than 6')     
#         #     cv2.imwrite('{}/character-recognition-test/art2/{}'.format(base_path,name),a)
#         #     a=''
#         # elif(len(platetext)>7 and a!=''):
#         #     print(name)
#         #     cv2.imwrite('{}/character-recognition-test/art3/{}'.format(base_path,name),a)
#         #     a=''
#         # else: 
#         #     print (name)
#          #   a=''


#         # for i in range(0,len(chars)):
            
#         #     platetext='{}{}'.format(platetext,chars[i][0])

#         #     platescore.append(str(chars[i][1]*100)[:5])
#         #     result={'licenseplate':platetext , "score":platescore}


#         # label_id_offset = 1
#         # image_np_with_detections = image_np.copy()


#         # a=viz_utils.visualize_boxes_and_labels_on_image_array(
#         #             image_np_with_detections,
#         #             detections['detection_boxes'],
#         #             detections['detection_classes']+label_id_offset,
#         #             detections['detection_scores'],
#         #             category_index,
#         #             use_normalized_coordinates=True,
#         #             max_boxes_to_draw=20,
#         #             min_score_thresh=.90,
#         #             agnostic_mode=False)
#         # # cv2.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
#         # cv2.imshow('a',a)
#         # img2 = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
#         # im_pil = Image.fromarray(img2)
#         # im_pil.show()
#         # cv2.waitKey(0) 
#         # cv2.destroyAllWindows() 
    



