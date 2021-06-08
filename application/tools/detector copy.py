from tools.utils import label_map_util
import os
import cv2
class Detector:
    def __init__(self,min_confidence):
        self.min_confidence=min_confidence
        self.classes_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..', 'classes'))
        self.base_path=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..', 'received'))
    
    


    
    def detect_plate(self,classes,boxes,scores,height,width,image_path,image):
       

     
        plate_label_map = label_map_util.load_labelmap('{}/plate_localization.pbtxt'.format(self.classes_path))
        plate_categories = label_map_util.convert_label_map_to_categories(plate_label_map, max_num_classes=80, use_display_name=True)
        plate_category_index = label_map_util.create_category_index(plate_categories)

        chars=[]
        platetext=''
        charbox=[]
        charbox.append("[ymin, xmin, ymax, ymax]")
        platescore=[]
        result=[]
        crop_img=''
        result=platetext,platescore,charbox,crop_img

        for i,platebox in enumerate(boxes):
        #check online if confidence is >.0%
            if(scores[i]>self.min_confidence and plate_category_index[classes[i]]['name']=='plate'):
            
                class_name = plate_category_index[classes[i]]['name']
                accuracy=scores[i]
                ymin=round(boxes[i][0]*height)
                xmin=round(boxes[i][1]*width)
                ymax=round(boxes[i][2]*height)
                xmax=round(boxes[i][3]*width)
                char=[class_name,scores[i],ymin,xmin,ymax,xmax]
                chars.append(char)
                if(class_name=='plate'):
                    crop_img = image[ymin:ymax, xmin:xmax]
                    cv2.imwrite('{}/cropped-{}'.format(self.base_path,image_path),crop_img)
    
                chars = sorted(chars, key=lambda x: x[3])
                print(chars)
                for i in range(0,len(chars)):
                    platetext='{}{}'.format(platetext,chars[i][0])
                    charbox.append(str(chars[i]))
                    platescore.append(str(chars[i][1]*100)[:5])
                    result=platetext , platescore,charbox

                result=platetext,platescore,charbox,crop_img
          
       
        return result



       
   
        
    def detect_registration(self,classes,boxes,scores,height,width,image_path):
        character_label_map = label_map_util.load_labelmap('{}/character_recognition.pbtxt'.format(self.classes_path))
        character_categories = label_map_util.convert_label_map_to_categories(character_label_map, max_num_classes=90, use_display_name=True)
        character_category_index = label_map_util.create_category_index(character_categories)

      

        object_name = []
        object_score = []
        ymin,xmin,ymax,xmax=0,0,0,0
        chars=[]
        platetext=''
        charbox=[]
        charbox.append("[ymin, xmin, ymax, ymax]")
        platescore=[]
        previous_char=''
        count=0

        for i,platebox in enumerate(boxes):
        
            #check online if confidence is >30%
            if(scores[i]>self.min_confidence):
                class_name = character_category_index[classes[i]]['name']
                accuracy=scores[i]
                ymin=round(boxes[i][0]*height)
                xmin=round(boxes[i][1]*width)
                ymax=round(boxes[i][2]*height)
                xmax=round(boxes[i][3]*width)
                char=[class_name,scores[i],ymin,xmin,ymax,xmax]
                chars.append(char)
    
        chars = sorted(chars, key=lambda x: x[3])
        box1StartY, box1StartX, box1EndY, box1EndX=0,0,0,0
        previous_index=''
        achars=''

        characteroverlapindex=[]
        characternonoverlapindex=[]
        characteroverlapindexstring=''
        
        for i in range(0,len(chars)):
            (box1StartY, box1StartX, box1EndY, box1EndX) = box1StartY, box1StartX, box1EndY, box1EndX
            (box2StartY, box2StartX, box2EndY, box2EndX) = chars[i][2],chars[i][3],chars[i][4],chars[i][5]
            xA = max(box2StartX, box1StartX)
            yA = max(box2StartY, box1StartY)
            xB = min(box2EndX, box1EndX)
            yB = min(box2EndY, box1EndY)

            # if the boxes are intersecting, then compute the area of intersection rectangle
            if xB > xA and yB > yA:
    
                interArea = (xB - xA) * (yB - yA)
                    
            else:
                interArea = 0.0
            
            box1Area = (box1EndY - box1StartY) * (box1EndX - box1StartX)
            box2Area = (box2EndY - box2StartY) * (box2EndX - box2StartX)

            # compute the intersection area / box1 area
        
            iou = interArea / float(box1Area + box2Area - interArea)
            if(iou>=0.8):
                count=count+1
                current_index=i
              
                print('the current char {} is intersection with {}: iou is {} index {} and {}'.format(previous_char,chars[i][0],iou,previous_index,current_index))
                characteroverlapindex.append(previous_index)
                characteroverlapindex.append(i)
                
        
            platetext='{}{}'.format(platetext,chars[i][0])
            
            characternonoverlapindex.append(previous_index)
            characternonoverlapindex.append(i)
                
            
            achars='{}{}'.format(achars,chars[i][0])
            # platetext='{}{}'.format(platetext,chars[i][0])
            charbox.append(str(chars[i][0:]))
            platescore.append(str(chars[i][1]*100)[:5])
            (box1StartY, box1StartX, box1EndY, box1EndX)=(box2StartY, box2StartX, box2EndY, box2EndX)
            previous_char=chars[i][0]
            previous_index=i
            
        #list all possible plate combination by detection the minimum intersection area

        characteroverlapindex=list(set(characteroverlapindex))
     

        return (platetext,platescore,charbox,characteroverlapindex)

    def top_registration(self,characteroverlapindex,registrationtext,registrationscore):
        
        reg_filter=characteroverlapindex
        reg_filter.sort()
        char=list()
        char.append(reg_filter[0:2])

        for i in range(2,len(reg_filter)):
            if(reg_filter[i-1]+1==reg_filter[i]):
                if(reg_filter[i-1]!=char[0][-1:][0]):
                    char[0].append('/')
                    char[0].append(reg_filter[i-1])
                char[0].append(reg_filter[i])
     
        char[0].append('/')   
        for i in range(0,len(char[0])):
            if(char[0][i]=='/'):
                print('hello')
        test_list=char[0]
     
        size = len(test_list)
        idx_list = [idx + 1 for idx, val in  enumerate(test_list) if val == '/']
        res = [test_list[i: j-1] for i, j in   zip([0] + idx_list, idx_list + ([size] if idx_list[-1] != size else []))]
        
        def replace_char_at_index(org_str, character_index, replacement):

            new_str = org_str
            if character_index < len(org_str):
                new_str = org_str[0:character_index] + replacement + org_str[character_index + 1:]
            return new_str


      
        plate_registration = registrationtext
        character_index=''
        top_registration=[]
        for i in range(0,len(plate_registration)):
            character_index='{}{}'.format(character_index,i)
        # Index positions
        separator='$/#%'
        testplate=[]
        if len(res)>=2:
            for i in range(0,len(res)):
                list_of_indexes=res[i]
                for j in list_of_indexes:
                    character_index = replace_char_at_index(character_index, j, separator[i])
            character_index=''.join(sorted(set(character_index), key=character_index.index))


            for n in res[0]:
                testplate.append( character_index.replace('$',str(n))) 
            for n in res[1]:
                for i in range(0,len(testplate)):
                    top_registration.append(testplate[i].replace('/',str(n)))
            # for n in res[1]:
            #     for i in range(0,len(testplate)):
            #         top_registration.append(testplate[i].replace('#',str(n)))
            
       

        else:   
            list_of_indexes =res[0] 
         
            # Replace characters at index positions in list
            for j in list_of_indexes:
                character_index = replace_char_at_index(character_index, j, '$')
            character_index=''.join(sorted(set(character_index), key=character_index.index))  
       
            for n in list_of_indexes:
                top_registration.append(character_index.replace('$',str(n)))
     

        def Convert(string):
            list1=[]
            list1[:0]=string
            return list1
    

        def generate_plates(A,B,C):
            registration=''
            confidence_level=0
            tp=list()
         
            for n in A:
                registration='{}{}'.format(registration,B.get(int(n)))
            tp.append(registration)
            for k in A: 
                
                confidence_level=confidence_level+float(C.get(int(k)))
            confidence_level=confidence_level/len(A)
            tp.append(confidence_level)
       
            return tp


        def Convert(string):
            list1=[]
            list1[:0]=string
            return list1


        t=[]
        for i in range(0,len(top_registration)):
            A=Convert(top_registration[i])
            B=(dict(list(enumerate(plate_registration))))
            C=(dict(list(enumerate(registrationscore))))
            z=generate_plates(A,B,C)
            t.append(z)

        result = sorted(t, key=lambda x: x[1],reverse=True)
        return result
   
            


    def detect_state(self):
        print('juris')

if __name__ == '__main__':
    print('main')
  


