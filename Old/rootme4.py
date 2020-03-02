from PIL import Image
from pytesser import *
im = Image.open('wjNL6.jpg')
text = image_to_string(im)
print(text)