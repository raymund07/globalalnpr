from tools.detector import Detector
from flask import Flask, render_template, request, jsonify
from numpy.testing._private.utils import print_assert_equal
from werkzeug import secure_filename
import base64
import io
from PIL import Image
from inference import Inference
import shutil
import os
from statistics import mean
import time

application = Flask(__name__)
base_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'received'))

@application.route('/upload')
def upload_render():

   return render_template('main/template/index-download.html')
@application.route('/web')
def upload_web():
   return render_template('webupload.html')

@application.route('/api/v2' , methods=['POST'])

def upload_apiv2():
  start_time = time.time()
  file = request.files['image']
  filename = secure_filename(file.filename)
  file.save('{}/{}'.format(base_path,filename))
  image_uploded=Inference()
  roi=Detector(0.30)
  image_path=filename
  print(image_path)
  classes,boxes,scores,height,width,image=list(image_uploded.predict_plate(filename))
  curTime = time.time()
  processingTime = curTime - start_time
  platelabel,platescore,platebox,crop_image=roi.detect_plate(classes,boxes,scores,height,width,image_path,image)
  plate_result={"plate":{"platelabel":platelabel, "platescore":platescore,"platebox":platebox,"imagename":image_path}}
  if (platelabel=='plate'):
      curTime = time.time()
      image_cropped=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'received'))
      classes,boxes,scores,height,width=image_uploded.predict_registration('{}/cropped-{}'.format(image_cropped,image_path))
      registrationlabel,registrationscore,registrationbox,registrationoverlapindex=roi.detect_registration(classes,boxes,scores,height,width,image_cropped)
      processingTime = curTime - start_time
      topregistration=roi.top_registration(registrationoverlapindex,registrationlabel,registrationscore)
      if(topregistration==[]):
        labeltext=registrationlabel
        registrationscore=list(map(float, registrationscore))
        if(len(registrationscore)>=1):
          registrationlabel=[labeltext,round(mean(registrationscore),2)]
      else:
        registrationlabel=topregistration[0]
      registration_result={"registration":{"processingTime":processingTime,"registrationlabel":registrationlabel, "registrationscore":registrationscore,"registrationbox":registrationbox,"imagename":image_path,"top_registration":topregistration}}
  else:
      classes,boxes,scores,height,width=image_uploded.predict_registration('{}'.format(image_path))
      registrationlabel,registrationscore,registrationbox,registrationoverlapindex=roi.detect_registration(classes,boxes,scores,height,width,image_path)
      processingTime = curTime - start_time
      topregistration=roi.top_registration(registrationoverlapindex,registrationlabel,registrationscore)
      if(topregistration==[]):
        labeltext=registrationlabel
        registrationscore=list(map(float, registrationscore))
        if(len(registrationscore)>=1):
          registrationlabel=[labeltext,round(mean(registrationscore),2)]
      else:
        registrationlabel=topregistration[0]

      registration_result={"registration":{"processingTime":processingTime,"registrationlabel":registrationlabel, "registrationscore":registrationscore,"registrationbox":registrationbox,"imagename":image_path,"top_registration":topregistration}}


  return jsonify(plate_result,registration_result)





@application.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
    shutil.rmtree("received")
    os.mkdir("received")
    #f = request.files['file']
    files = request.files.getlist("file[]")
   

  
    for f in files:
      #file = request.files.get(f)
      imageFile = "received/" + secure_filename(f.filename)

      f.save(imageFile)

    objectDetectResults = predictImages ()
    return jsonify(objectDetectResults)


if __name__ == '__main__':
  # application.run(host='0.0.0.0',port=5000,debug = True)
   application.run(port=5000,debug = True)



