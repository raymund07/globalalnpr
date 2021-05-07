from flask import Flask, render_template, request, jsonify
from werkzeug import secure_filename
import base64
import io
from PIL import Image
from predict_images import predictImages
import shutil
import os
import subprocess
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

application = Flask(__name__)

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
  #file.save('received/{}'.format(file.name))
  file.save('uploadedImages/{}'.format(filename))
  # # Read the image via file.stream
  img = Image.open(file.stream)
  objectDetectResults = predictImages()
  return jsonify(objectDetectResults)


@application.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
    shutil.rmtree("uploadedImages")
    os.mkdir("uploadedImages")
    #f = request.files['file']
    files = request.files.getlist("file[]")
   

  
    for f in files:
      #file = request.files.get(f)
      imageFile = "uploadedImages/" + secure_filename(f.filename)

      f.save(imageFile)
    modelArg, labelsArg, imagePathArg, num_classesArg, min_confidenceArg, image_displayArg, pred_stagesArg
    
    objectDetectResults = predictImages ()
    return jsonify(objectDetectResults)


if __name__ == '__main__':
   application.run(port=5000,debug = True)


