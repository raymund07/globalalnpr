import os
import requests
b=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'images'))

# # my_img = {'image': open('{}/2.jpg'.format(b), 'rb')}
# # r = requests.post('http://localhost:5000/api', files=my_img,data={'model':'character'})
# # print(r.text)

# for i in os.listdir(b):


#     my_img = {'image': open('{}/{}'.format(b,i), 'rb')}
#     r = requests.post('http://localhost:5000/api/v2', files=my_img, data={'model':'character'})
#     print(r.text)

my_img = {'image': open('{}/7.jpg'.format(b), 'rb')}
r = requests.post('http://localhost:5000/api/v2', files=my_img, data={'model':'plate'})
print(r.text)


# my_img = {'image': open('{}/6.jpg'.format(b), 'rb')}
# r = requests.post('http://localhost:5000/api/v2', files=my_img, data={'model':'plate'})


import json
y=json.loads(r.text)
string=y[1]['registration']['character']
index=y[1]['registration']['overlap']
print(y[1]['registration']['character'])
print(y[1]['registration']['overlap'])

# print(type(index))
# print(len(index))

# import itertools
# next_list=list(itertools.product(y[1]['registration']['character'],y[1]['registration']['overlap']))
# print(next_list)

l ='123459679'
filter1=index
filter1.sort()
filter2=[0,1,6]
char=[]
char.append(filter1[0:2])
for i in range(2,len(filter1)):
    if(filter1[i-1]+1==filter1[i]):
        if(filter1[i-1]!=char[0][-1:][0]):
            char[0].append('/')
            char[0].append(filter1[i-1])
        char[0].append(filter1[i])
        

        
char[0].append('/')   
for i in range(0,len(char[0])):
    if(char[0][i]=='/'):
        print('hello')
test_list=char[0]
print(test_list)
size = len(test_list)
idx_list = [idx + 1 for idx, val in
            enumerate(test_list) if val == '/']
res = [test_list[i: j-1] for i, j in
        zip([0] + idx_list, idx_list + 
        ([size] if idx_list[-1] != size else []))]

def replace_char_at_index(org_str, character_index, replacement):
    ''' Replace character at index in string org_str with the
    given replacement character.'''
    new_str = org_str
    if character_index < len(org_str):
        new_str = org_str[0:character_index] + replacement + org_str[character_index + 1:]
    return new_str


print(res)

print(len(res))
plate_registration = y[1]['registration']['character']
character_index=''
for i in range(0,len(plate_registration)):
    character_index='{}{}'.format(character_index,i)
# Index positions
separator='$/#%'
print(character_index)
if len(res)>1:
    for i in range(0,len(res)):
        list_of_indexes=res[i]
        for j in list_of_indexes:
            character_index = replace_char_at_index(character_index, j, separator[i])
    character_index=''.join(sorted(set(character_index), key=character_index.index))    
    print(character_index)
    for i in range (0,len(res)):
        print(character_index.replace(separator[i]))
        # for n in res[i]:
        #     print(character_index.replace(separator[i],str(n)))
      
else:   
    list_of_indexes =res 
    # Replace characters at index positions in list
    for j in list_of_indexes:
        character_index = replace_char_at_index(character_index, j, '$')
    character_index=''.join(sorted(set(character_index), key=character_index.index))   
    for i in range(0,res):
        print(i)
        # for n in res[i]:
        #     print(n)
        #     # print(character_index.replace(separator[i],str(n)))



# a = list(filter(None, sample_str.split('/')))
# # list(set(characteroverlapindex))




# string='gwu$ab/'
# for n in b[0]:
#   print(string.replace('$',str(n)))
# for n in b[1]:
#   print(string.replace('/',str(n)))
# for n in b[1]:
#   print(string.replace('/',str(n)))

# sample_str='012345678'
# index_string=''
# for i in range(0,len(sample_str)):
#     index='{}{}'.format(index,i)
# print(index)

# list_of_indexes = [4,5,6]
# # Replace characters at index positions in list
# for index in list_of_indexes:
#     sample_str = replace_char_at_index(sample_str, index, '/')

# print(sample_str)