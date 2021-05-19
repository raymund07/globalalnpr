import pydbgen  
from pydbgen import pydbgen
from PIL import Image, ImageDraw, ImageFont
myDB=pydbgen.pydb()
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import random
from random import randrange
import os
import time


def generate_texas_2012(base_path,number_of_images):

     texas_2009=list()
     texas_2007=list()
     myDB.city_real()
     for i in range (number_of_images):
          a=myDB.license_plate(seed=3)
          if(a[3]=='-' and len(a)==8):
               texas_2009.append(a)
          if(a[3]=='-' and len(a)==7):
               texas_2007.append(a)

     for n in texas_2007:
          img = Image.open('bg/tx2009-rgb.jpg')
          img = img.convert('RGB')
          d1 = ImageDraw.Draw(img)
          myFont = ImageFont.truetype('font/california2.ttf', 48)
          d1.text((7, 21), n[0] , font=myFont, fill =(0,0,0))
          d1.text((9, 21), n[0] , font=myFont, fill =(0,0,0))

          d1.text((28, 21), n[1] , font=myFont, fill =(0,0,0))
          d1.text((29, 21), n[1] , font=myFont, fill =(0,0,0))

          d1.text((48, 21), n[2] , font=myFont, fill =(0,0,0))
          d1.text((49, 21), n[2] , font=myFont, fill =(0,0,0))

          d1.text((78, 21), n[4] , font=myFont, fill =(0,0,0))
          d1.text((79, 21), n[4] , font=myFont, fill =(0,0,0))

          d1.text((99, 21), n[5] , font=myFont, fill =(0,0,0))
          d1.text((100, 21), n[5] , font=myFont, fill =(0,0,0))

          d1.text((119, 21), n[6] , font=myFont, fill =(0,0,0))
          d1.text((120, 21), n[6] , font=myFont, fill =(0,0,0))

          img.save('{}/{}.jpg'.format(base_path,n))

def generate_fl(base_path,number_of_images):
     f1=list()
     f2=list()
     for i in range (number_of_images):
          a=myDB.license_plate(seed=3)
          if(a[3]!='-' and len(a)==7):
               f1.append(a)
          if(a[3]=='-' and len(a)==7):
               f2.append(a)


     for n in f2:
          img = Image.open('bg/fl2004x150.jpg')
          img = img.convert('RGB')
          myFont = ImageFont.truetype('font/dealerplate_florida.otf', 40)
          d1 = ImageDraw.Draw(img)
         
             
          d1.text((8, 22), n[0] , font=myFont, fill =(17,125,99))
          d1.text((25, 22), n[1] , font=myFont, fill =(17,125,99))
          d1.text((41, 22), n[2] , font=myFont, fill =(17,125,99))
 
          d1.text((90, 22), n[4] , font=myFont, fill =(17,125,99))
          d1.text((107, 22), n[5] , font=myFont, fill =(17,125,99))
          d1.text((124, 22), n[6] , font=myFont, fill =(17,125,99))
     
     
          img.save('{}/{}.jpg'.format(base_path,n))

def generate_ca_2011(a):

     ca_2011=list()
     ca_1980=list()
     myDB.city_real()
     for i in range (a):
          a=myDB.license_plate(seed=3)
          if(a[3]!='-' and len(a)==7):
               ca_2011.append(a)
          if(a[3]=='-' and len(a)==7):
               ca_1980.append(a)

     for n in ca_2011:
          time.sleep(2)
          img = Image.open('bg/CA-2011.jpg')
          img = img.convert('RGB')
          d1 = ImageDraw.Draw(img)
          myFont = ImageFont.truetype('font/california.ttf', 42)
               
          d1.text((10, 30), n[0] , font=myFont, fill =(23,29,72))
          d1.text((30, 30), n[1] , font=myFont, fill =(23,29,72))
          d1.text((50, 30), n[2] , font=myFont, fill =(23,29,72))
          d1.text((68, 30), n[3] , font=myFont, fill =(23,29,72))
          d1.text((86, 30), n[4] , font=myFont, fill =(23,29,72))
          d1.text((104, 30), n[5] , font=myFont, fill =(23,29,72))
          d1.text((124, 30), n[6] , font=myFont, fill =(23,29,72))
     

          img.save('images/california/data/{}.jpg'.format(n))
def covert_to_gray(): 
     for i in os.listdir('images/california/data'):

          image = cv2.imread('images/california/data/{}'.format(i))
          image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          cv2.imwrite('images/california/data/{}'.format(i),image)

def draw_box(base_path):
      for i in os.listdir(base_path):
           #californial bounding box
          image = cv2.imread('{}/{}'.format(base_path,i))
          # image = cv2.rectangle(image, (10,28), (28,63), (255,0,0), 1)
          # image= cv2.rectangle(image, (30,28), (48,63), (255,0,0), 1)
          # image= cv2.rectangle(image, (50,28), (68,63), (255,0,0), 1)
          # image= cv2.rectangle(image, (68,28), (86,63), (255,0,0), 1)
          # image= cv2.rectangle(image, (86,28), (104,63), (255,0,0), 1)
          # image= cv2.rectangle(image, (104,28), (122,63), (255,0,0), 1)
          # image= cv2.rectangle(image, (123,28), (141,63), (255,0,0), 1)
         
          #texas 2006

          # image = cv2.rectangle(image, (10,20), (28,57), (255,0,0), 1)
          # image= cv2.rectangle(image, (31,20), (48,57), (255,0,0), 1)
          # image= cv2.rectangle(image, (52,20), (69,57), (255,0,0), 1)
          # image= cv2.rectangle(image, (81,20), (98,57), (255,0,0), 1)
          # image= cv2.rectangle(image, (102,20), (119,57), (255,0,0), 1)
          # image= cv2.rectangle(image, (122,20), (139,57), (255,0,0), 1)

          image = cv2.rectangle(image, (8,21), (25,57), (255,0,0), 1)
          image= cv2.rectangle(image, (25,21), (42,57), (255,0,0), 1)
          image= cv2.rectangle(image, (41,21), (58,57), (255,0,0), 1)
          image= cv2.rectangle(image, (90,21), (107,57), (255,0,0), 1)
          image= cv2.rectangle(image, (107,21), (124,57), (255,0,0), 1)
          image= cv2.rectangle(image, (124,21), (141,57), (255,0,0), 1)
   
   


          cv2.imwrite('{}/{}'.format(base_path,i),image)



     
def preprocess(base_path):
     for i in os.listdir(base_path):
    
          image = cv2.imread('{}/{}'.format(base_path,i))

          #texas classic 2012 
          # image = cv2.rectangle(image, (12,30), (27,63), (255,0,0), 1)
          # image= cv2.rectangle(image, (30,30), (45,63), (255,0,0), 1)
          # image= cv2.rectangle(image, (46,30), (61,63), (255,0,0), 1)
          # image= cv2.rectangle(image, (81,30), (96,63), (255,0,0), 1)
          # image= cv2.rectangle(image, (98,30), (113,63), (255,0,0), 1)
          # image= cv2.rectangle(image, (117,30), (132,63), (255,0,0), 1)
          # image= cv2.rectangle(image, (134,30), (149,63), (255,0,0), 1)

          #texas classic 2009
          # image = cv2.rectangle(image, (9,23), (25,55), (255,0,0), 1)
          # image= cv2.rectangle(image, (26,23), (42,56), (255,0,0), 1)
          # image= cv2.rectangle(image, (45,23), (58,56), (255,0,0), 1)
          # image= cv2.rectangle(image, (75,23), (91,56), (255,0,0), 1)
          # image= cv2.rectangle(image, (92,23), (108,56), (255,0,0), 1)
          # image= cv2.rectangle(image, (109,23), (126,56), (255,0,0), 1)
          # image= cv2.rectangle(image, (127,23), (142,56), (255,0,0), 1)

          c=np.ones(image.shape, dtype="uint8") * (randrange(10,60))
          r=randrange(0,2)
          if (r==1):
               a=cv2.add(image,c)
         
               cv2.imwrite('{}/{}'.format(base_path,i),a)
          else:
               b=cv2.subtract(image,c)
      
               cv2.imwrite('{}/{}'.format(base_path,i),b)




if __name__ == '__main__':
   
     base_path='images/florida/2004'
     number_of_images=5000
     # generate_texas_2012(base_path,number_of_images)
     # generate_ca_2011(50000)
     generate_fl(base_path,number_of_images)
     # preprocess(base_path)
     # covert_to_gray()
     # draw_box(base_path)
     

