from flask import Flask, render_template, request, jsonify
from werkzeug import secure_filename
import base64
import io
from PIL import Image

from predict_multiplegraph import plate
from predict_multiplegraph import character
from predict_multiplegraph import jurisdiction
import shutil
import os
import subprocess
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

application = Flask(__name__)
base_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'received'))

@application.route('/upload')
def upload_render():

   return render_template('main/template/index-download.html')
@application.route('/web')
def upload_web():
   return render_template('webupload.html')

@application.route('/api' , methods=['POST'])
def upload_api():
  file = request.files['image']
  filename = secure_filename(file.filename)
  #file.save('{}/{}'.format(file.name))
  file.save('{}/{}'.format(base_path,filename))
  # # Read the image via file.stream
  img = Image.open(file.stream)
  objectDetectResults = predictImages()
  return jsonify(objectDetectResults)

@application.route('/api/v2' , methods=['POST'])

def upload_apiv2():
  charDetectResults=[]
  objectDetectResults=[]
  file = request.files['image']
  filename = secure_filename(file.filename)
  file.save('{}/{}'.format(base_path,filename))
  plateDetectResults = plate(filename)
  # detect if plate is available and crop to determine characters
  if(plateDetectResults['plate']['label']=='plate'):
    charDetectResults=character('cropped-{}'.format(filename))
  else:
    charDetectResults=character('{}'.format(filename))
  return jsonify(plateDetectResults,charDetectResults)

@application.route('/api/v2/test' , methods=['POST'])
def upload_apiv2test():
  test=request.files
  test=request.form['model']
  print(test)
  return test



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
    modelArg, labelsArg, imagePathArg, num_classesArg, min_confidenceArg, image_displayArg, pred_stagesArg
    
    objectDetectResults = predictImages ()
    return jsonify(objectDetectResults)


if __name__ == '__main__':
   application.run(port=5000,debug = True)


