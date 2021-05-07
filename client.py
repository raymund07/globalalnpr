import os
import requests
my_img = {'image': open('images/test.jpg', 'rb')}
r = requests.post('http://localhost:5000/api', files=my_img)
print(r.text)


