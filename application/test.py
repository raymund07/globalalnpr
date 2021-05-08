import os
import requests
import sys
import io
from PIL import Image
from predict_multiplegraph import plate
from predict_multiplegraph import character
from predict_multiplegraph import jurisdiction

b=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'Inferencegraphs'))
im = Image.open("{}/2.jpg".format(b))
a =plate("2.jpg")
print(a)

b=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'Inferencegraphs'))
im = Image.open("{}/2.jpg".format(b))
a =plate("2.jpg")
print(a)


b=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'Inferencegraphs'))
im = Image.open("{}/2.jpg".format(b))
a =plate("2.jpg")
print(a)


