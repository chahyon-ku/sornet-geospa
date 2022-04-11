import io
import os

import numpy
from PIL import Image
import h5py


with h5py.File('data/geospa/objects.h5', 'w') as f:
    views_dir = '../views_geospa/'
    for root, dirs, files in os.walk(views_dir):
        if len(dirs) > 0:
            continue
        object_name = root[16:]
        print(object_name, files)
        img_bytes_array = []
        for i, file in enumerate(files):
            img_pil = Image.open(root + '/' + file)
            img_pil = img_pil.convert('RGB')
            img_pil = img_pil.resize((32, 32))
            buf = io.BytesIO()
            img_pil.save(buf, 'png')
            img_bytes = buf.getvalue()
            img_bytes = numpy.array(img_bytes)
            img_bytes_array.append(img_bytes)
        #print(img_bytes_array)
        f.create_dataset(object_name, shape=(len(files), ), data=img_bytes_array)