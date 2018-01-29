import numpy as np
from PIL import Image
import random

im = Image.open("/home/rokas/Workspace/school/simulation/subscribe/datasets/DM-RECT-200-V2/1_map_nw_16_1_20100630.tif")

width = im.width
height = im.height

frame_width = 640
frame_height = 480

for i in range(167):
  x = random.randint(50, width - frame_width - 50)
  y = random.randint(50, height - frame_height - 50)

  xn = x
  while abs(xn - x) < frame_width:
    xn = random.randint(50, width - frame_width - 50)

  yn = y
  while abs(yn - y) < frame_height:
    yn = random.randint(50, height - frame_height - 50)

  im1 = im.crop((x-64, y-48, x+frame_width+64, y+frame_height+48))

  x = x + random.randint(0, 10)
  y = y + random.randint(0, 10)

  im2 = im.crop((x-64, y-48, x+frame_width+64, y+frame_height+48))

  im2 = im2.rotate(random.randint(0, 10))


  im3 = im.crop((xn-64, yn-48, xn+frame_width+64, yn+frame_height+48))

  im1 = im1.crop((64, 48, 640+64, 480+48))
  im2 = im2.crop((64, 48, 640+64, 480+48))
  im3 = im3.crop((64, 48, 640+64, 480+48))

#  im1.save("negatives/im-a-" + str(i) + ".jpg")
#  im2.save("negatives/im-b-" + str(i) + ".jpg")
  im3.save("negatives/im-neg-" + str(i) + ".jpg")

