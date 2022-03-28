
from flask import Flask, render_template, request, jsonify
from numpy.testing._private.utils import print_assert_equal
from werkzeug.utils import secure_filename
import base64
import io
from PIL import Image
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

@application.route('/api/v2' , methods=['POST'])
def upload_apiv2():
  start_time = time.time()
  # file = request.files['image']
  
  confidence=request.form['confidence']
  version=request.form['version']
  print(confidence,version)

  # filename = secure_filename(file.filename)
  # file.save('{}/{}'.format(base_path,filename))
  

  return jsonify()





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
  http = WSGIServer(('', 5000), application.wsgi_app) 

    # Serve your application
  http.serve_forever()



