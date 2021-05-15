from globalhelper.utils import label_map_util


class Inference:
    def __init__(self):
        a='h'
        print('hello')
    
    def detect_plate(self):
        print('plate')
   
        
    def detect_char(self):
        print('char')
        print(self)

    def detect_state(self):
        print('juris')

if __name__ == '__main__':
    Inference()
  
# import numpy as np

# class Helper:
  
#   def __init__(self, minConfidence, categoryIdx, charIOUMax=0.3, charPlateIOAMin=0.5, rejectPlates=False, minScore=0.6, minChars=2):
#     # boxes below minConfidence are rejected
#     self.minConfidence = minConfidence
#     # character boxes that do not overlap plate box by at least 'charPlateIOAMin' are rejected
#     self.charPlateIOAMin = charPlateIOAMin
#     # character boxes overlapping other char boxes by more than 'charIOUMax' are rejected
#     self.charIOUMax = charIOUMax
#     # If 'rejectPlates' is True, then plates with a "complete score" less than minScore,
#     # or less than 'minChars' characters, or plate boxes that touch the edge of the frame, will be rejected
#     self.rejectPlates = rejectPlates
#     self.minScore = minScore
#     self.minChars = minChars
#     self.categoryIdx = categoryIdx

#  def intersectionOverUnion(self, , box2):
#     (box1StartY, box1StartX, box1EndY, box1EndX) = box1
#     (box2StartY, box2StartX, box2EndY, box2EndX) = box2
#     # determine the (x, y)-coordinates of the intersection rectangle
#     xA = max(box2StartX, box1StartX)
#     yA = max(box2StartY, box1StartY)
#     xB = min(box2EndX, box1EndX)
#     yB = min(box2EndY, box1EndY)

#     # if the boxes are intersecting, then compute the area of intersection rectangle
#     if xB > xA and yB > yA:
#       interArea = (xB - xA) * (yB - yA)
#     else:
#       interArea = 0.0

#     # compute the area of the box1 and box2
#     box1Area = (box1EndY - box1StartY) * (box1EndX - box1StartX)
#     box2Area = (box2EndY - box2StartY) * (box2EndX - box2StartX)

#     # compute the intersection area / box1 area
#     iou = interArea / float(box1Area + box2Area - interArea)

#     # return the intersection over area value
#     return iou


#   import json
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
#     for i in range(0,len(A)):
#         registration='{}{}'.format(registration,B.get(i))
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

