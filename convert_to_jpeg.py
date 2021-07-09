import glob
from PIL import Image
import os.path
foo = input('경로: ')
files = glob.glob('/home/hong/PycharmProjects/pythonProject/alexNet/input/task/train/'+foo+'/*')

for item in files:
    fname = os.path.splitext(item)
    os.rename(item, fname[0].replace(' ','_'))

files = glob.glob('/home/hong/PycharmProjects/pythonProject/alexNet/input/task/train/' + foo + '/*')
for item in files:
    if item.endswith('.jpeg'):
        continue
    else:
        try:
            im = Image.open(item)
            bg = Image.new("RGB", im.size, (255,255,255))
            bg.paste(im,im)
            fname = os.path.splitext(item)
            bg.save(fname[0]+'.jpeg')
            os.remove(item)
        except:
            # os.remove(item)
            fname = os.path.splitext(item)
            os.rename(item, fname[0] + '.jpeg')
            # os.remove(item)
            continue
            # fname = os.path.splitext(item)
            # try:
            #     os.rename(item, fname[0] + '.jpeg')
            # except:
            #     pass

