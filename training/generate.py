import pydbgen  
from pydbgen import pydbgen
from PIL import Image, ImageDraw, ImageFont
myDB=pydbgen.pydb()


texas_2009=list()
texas_2007=list()
myDB.city_real()
for i in range (1000):
    a=myDB.license_plate(seed=3)
    if(a[3]=='-' and len(a)==8):
        texas_2009.append(a)
    if(a[3]=='-' and len(a)==7):
        texas_2009.append(a)

 



print(texas_2009)
print(texas_2007)


# from PIL import Image, ImageDraw, ImageFont

# img = Image.open('images/texas/2012/3.jpg')
# d1 = ImageDraw.Draw(img)
# myFont = ImageFont.truetype('images/texas/LicensePlate.ttf', 40)
# d1.text((12, 26), "ABC-1234", font=myFont, fill =(0, 0, 0))
# img.show()

# img.save("images/texas/2012/data/1.jpg")

    