import os
import requests
b=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'images'))

# my_img = {'image': open('{}/2.jpg'.format(b), 'rb')}
# r = requests.post('http://localhost:5000/api', files=my_img,data={'model':'character'})
# print(r.text)

for i in os.listdir(b):


    my_img = {'image': open('{}/{}'.format(b,i), 'rb')}
    r = requests.post('http://localhost:5000/api/v2', files=my_img,data={'model':'character'})
    print(r.text)

