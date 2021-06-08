import os
import requests
import json
b=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'images'))

c=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'training/images/texas'))

# my_img = {'image': open('{}/2.jpg'.format(b), 'rb')}
# r = requests.post('http://localhost:5000/api/v2', files=my_img,data={'model':'character'})
# print(r.text)


for i in os.listdir('F:/global/2019-04-02/2019-04-02-05-48-00_43-3964'):


    my_img = {'image': open('F:/global/2019-04-02/2019-04-02-05-48-00_43-3964/{}'.format(i), 'rb')}
    r = requests.post('http://localhost:5000/api/v2', files=my_img, data={'model':'character'})
    print(r.text)
    # y=json.loads(r.text)
    # print(y)
    # string=y[1]['registration']['character']
    # index=y[1]['registration']['overlap']
  





# my_img = {'image': open('{}/p3.jpg'.format(b), 'rb')}
# r = requests.post('https://globalalnpr.azurewebsites.net/api/v2', files=my_img, data={'model':'plate'})
# print(r.text)

# my_img = {'image': open('{}/7.JPG'.format(b), 'rb')}
# r = requests.post('http://localhost:5000/api/v2', files=my_img, data={'model':'plate'})
# print(r.text)
# import cv2

s=r'C:\Users\Isaac\global\tensorflow-anpr\received/cropped-0_1.jpg'
cv2.imread(r(string))


# import json
# y=json.loads(r.text)
# string=y[1]['registration']['character']
# index=y[1]['registration']['overlap']

# filter1=index
# filter1.sort()
# char=[]
# char.append(filter1[0:2])

# for i in range(2,len(filter1)):
#     if(filter1[i-1]+1==filter1[i]):
#         if(filter1[i-1]!=char[0][-1:][0]):
#             char[0].append('/')
#             char[0].append(filter1[i-1])
#         char[0].append(filter1[i])
        

        
# char[0].append('/')   
# for i in range(0,len(char[0])):
#     if(char[0][i]=='/'):
#         print('hello')
# test_list=char[0]
# print(test_list)
# size = len(test_list)
# idx_list = [idx + 1 for idx, val in
#             enumerate(test_list) if val == '/']
# res = [test_list[i: j-1] for i, j in
#         zip([0] + idx_list, idx_list + 
#         ([size] if idx_list[-1] != size else []))]

# def replace_char_at_index(org_str, character_index, replacement):
#     ''' Replace character at index in string org_str with the
#     given replacement character.'''
#     new_str = org_str
#     if character_index < len(org_str):
#         new_str = org_str[0:character_index] + replacement + org_str[character_index + 1:]
#     return new_str


# print(res)

# print(len(res))
# plate_registration = y[1]['registration']['character']
# character_index=''
# top_registration=[]
# for i in range(0,len(plate_registration)):
#     character_index='{}{}'.format(character_index,i)
# # Index positions
# separator='$/#%'
# print(character_index)
# testplate=[]
# if len(res)>=2:
#     for i in range(0,len(res)):
#         list_of_indexes=res[i]
#         for j in list_of_indexes:
#             character_index = replace_char_at_index(character_index, j, separator[i])
#     character_index=''.join(sorted(set(character_index), key=character_index.index))


#     for n in res[0]:
#         testplate.append( character_index.replace('$',str(n))) 
#     for n in res[1]:
#         for i in range(0,len(testplate)):
#             top_registration.append(testplate[i].replace('/',str(n)))
    
    
#     print(character_index)

      
# else:   
#     list_of_indexes =res[0] 
#     print(list)
#     # Replace characters at index positions in list
#     for j in list_of_indexes:
#         character_index = replace_char_at_index(character_index, j, '$')
#     character_index=''.join(sorted(set(character_index), key=character_index.index))  
#     print(character_index)
#     for n in list_of_indexes:
#         top_registration.append(character_index.replace('$',str(n)))
    
# print(top_registration)

# def Convert(string):
#     list1=[]
#     list1[:0]=string
#     return list1
# import pandas as pd

# def generate_plates(A,B):
#     registration=''
#     print(B.get(0))
#     for n in A:
#         registration='{}{}'.format(registration,B.get(int(n)))
#     return registration


# def Convert(string):
#     list1=[]
#     list1[:0]=string
#     return list1


# t=[]
# for i in range(0,len(top_registration)):
#     A=Convert(top_registration[i])
#     B=(dict(list(enumerate(plate_registration))))
#     z=generate_plates(A,B)
#     t.append(z)
# print(t) 
# print(B)

# import pandas as pd

