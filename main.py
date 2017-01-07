# coding=utf-8

from selective_search import selective_search
import skimage.data
from PIL import ImageDraw, Image

img = skimage.data.coffee()

regions = selective_search(img, similarities=('color', 'texture', 'size', 'fill'))

# draw the rectangle
im = Image.fromarray(img)
draw = ImageDraw.Draw(im)
for r in regions:
    if r[-1] < 2000:
        continue
    coordinates = [(y, x) for x in r[:2] for y in r[2:4]]
    draw.line((coordinates[0] + coordinates[1]), fill=128)
    draw.line((coordinates[0] + coordinates[2]), fill=128)
    draw.line((coordinates[1] + coordinates[3]), fill=128)
    draw.line((coordinates[2] + coordinates[3]), fill=128)
del draw

im.show()
im.save('example.png')






