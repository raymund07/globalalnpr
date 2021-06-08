import time
import os
import tensorflow as tf
import cv2
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

print('Loading model...', end='')
start_time = time.time()
inference_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'inferencegraphs'))



# Load saved model and build the detection function
detect_fn = tf.saved_model.load('C:/Users/Isaac/global/training_demo/exported-models/plate/saved_model/')
category_index = label_map_util.create_category_index_from_labelmap('C:/Users/Isaac/global/tensorflow-anpr/classes/plate_localization.pbtxt')
end_time = time.time()



import numpy as np
from PIL import Image
import warnings
IMAGE_PATHS='F:/training/2019-04-14-00-38-04_22-6041/TX01011181020190413222821752753010190413222821275_E.JPG'

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


image_np = load_image_into_numpy_array(IMAGE_PATHS)

input_tensor = tf.convert_to_tensor(image_np)
input_tensor = input_tensor[tf.newaxis, ...]

detections = detect_fn(input_tensor)
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
 
 
print(detections)
