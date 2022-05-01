from tools.detector import Detector
from flask import Flask, render_template, request, jsonify
from numpy.testing._private.utils import print_assert_equal
from werkzeug.utils import secure_filename
import base64
import io
from PIL import Image
from inference import Inference
import shutil
import os
from flask import Flask, Response
from gevent.pywsgi import WSGIServer
from gevent import monkey


from statistics import mean
import time
import os
application = Flask(__name__)
base_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'received'))

@application.route('/test', methods=['POST'])
def test():
   a= request.files['image']
   b= request.form['confidence']
   print(type(b))
   
   print(a,b)
   return 'hello'
@application.route('/api/character')
def plate():
    start_time = time.time()
    file = request.files['image']
    
    confidence=request.form['confidence']
    version=request.form['version']

    filename = secure_filename(file.filename)
    file.save('{}/{}'.format(base_path,filename))
    image_uploded=Inference()
    roi=Detector(float(confidence))
    image_path=filename
    result=image_uploded.predict_charlocation(image_path)
    return jsonify(result)

@application.route('/api/v2' , methods=['POST'])

def upload_apiv2():
  start_time = time.time()
  file = request.files['image']
  
  confidence=request.form['confidence']
  version=request.form['version']

  filename = secure_filename(file.filename)
  file.save('{}/{}'.format(base_path,filename))
  image_uploded=Inference()
  roi=Detector(float(confidence))
  image_path=filename
  classes,boxes,scores,height,width,image=list(image_uploded.predict_plate(filename))
  curTime = time.time()
  processingTime = curTime - start_time
  platelabel,platescore,platebox,crop_image=roi.detect_plate(classes,boxes,scores,height,width,image_path,image)
  plate_result={"plate":{"platelabel":platelabel, "platescore":platescore,"platebox":platebox,"imagename":image_path}}
  if (platelabel=='plate'):


      curTime = time.time()
      image_cropped=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'received'))
      classes,boxes,scores,height,width=image_uploded.predict_registration('{}/cropped-{}'.format(image_cropped,image_path))
      jurisdiction=image_uploded.predict_jurisdiction('{}/cropped-{}'.format(image_cropped,image_path))
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
      jurisdiction=image_uploded.predict_jurisdiction('{}/cropped-{}'.format(image_cropped,image_path))
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
      
      registration_result={"registration":{"processingTime":processingTime,"registrationlabel":registrationlabel,"registrationscore":registrationscore,"registrationbox":registrationbox,"imagename":image_path,"top_registration":topregistration}}
  
  os.remove('{}/{}'.format(base_path,filename))
  os.remove('{}/cropped-{}'.format(base_path,filename))
  print(registration_result,jurisdiction)


  return jsonify(plate_result,registration_result,jurisdiction)





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
  #  application.run(port=5000,debug = True)
  http = WSGIServer(('0.0.0.0', 5000), application.wsgi_app) 

    # Serve your application
  http.serve_forever()


