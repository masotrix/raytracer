#!/usr/bin/python3
import sys
from PIL import Image
if (len(sys.argv)<4):
    print("Usage: ./getPixel.py image x y");
    sys.exit();
im = Image.open(sys.argv[1]);
print (im.getpixel((int(sys.argv[2]),int(sys.argv[3]))))
im.close();
