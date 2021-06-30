import glob
from PIL import Image
import os.path
foo = input('경로: ')
files = glob.glob('/home/hong/PycharmProjects/pythonProject/alexNet/input/images/'+foo+'/*')

for item in files:
    if item.endswith('.jpeg'):
        # fname = os.path.splitext(item)
        # os.rename(item, fname[0].replace(' ',''))
        continue
    else:
        try:
            im = Image.open(item)
            bg = Image.new("RGB", im.size, (255,255,255))
            bg.paste(im,im)
            bg.save(item+'.jpeg')
            os.remove(item)
        except:
            # os.remove(item)
            fname = os.path.splitext(item)
            os.rename(item, fname[0] + '.jpeg')
            os.remove(item)
            continue
            # fname = os.path.splitext(item)
            # try:
            #     os.rename(item, fname[0] + '.jpeg')
            # except:
            #     pass

