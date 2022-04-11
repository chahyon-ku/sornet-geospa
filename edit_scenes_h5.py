import io
import os

import numpy
from PIL import Image
import h5py


# with h5py.File('data/geospa/objects.h5', 'w') as f:
#     views_dir = '../views/'
#     for root, dirs, files in os.walk(views_dir):
#         if len(dirs) > 0:
#             continue
#         object_name = root[9:]
#         print(object_name, files)
#         img_pil = Image.open(root + '/' + files[0])
#         img_pil = img_pil.convert('RGB')
#         img_pil = img_pil.resize((32, 32))
#         buf = io.BytesIO()
#         img_pil.save(buf, 'png')
#         img_bytes = buf.getvalue()
#         img_bytes = numpy.array(img_bytes)
#         f.create_dataset(object_name, shape=(len(files), ), data=img_bytes)


with h5py.File('data/geospa/sample2.h5', 'r') as f:
    with h5py.File('data/geospa/sample2_edit.h5', 'w') as w_f:
        for key in f:
            new_scene_name = key[-6:]
            print(new_scene_name)
            w_f.create_group(new_scene_name)

            image = f[key]['images'][next(iter(f[key]['images']))][()]
            image = Image.open(io.BytesIO(image))
            image = image.resize((480, 320))
            image_bytes = io.BytesIO()
            image.save(image_bytes, 'png')
            image_bytes = numpy.array(image_bytes.getvalue())
            w_f[new_scene_name].create_dataset('image', data=image_bytes)

            objects = [obj.replace(b' ', b'_') for obj in f[key]['objects']]
            objects = b','.join(objects)
            w_f[new_scene_name].create_dataset('objects', data=objects)

            relations = []
            for relation in f[key]['relations']:
                if relation in {'front', 'left', 'contain', 'support'}:
                    print(relation)
                    relation_np = numpy.array(f[key]['relations'][relation], dtype=numpy.int8)
                    numpy.fill_diagonal(relation_np, -1)
                    relations.append(relation_np)
            w_f[new_scene_name].create_dataset('relations', data=relations)
            print(w_f[new_scene_name]['relations'])

        print(w_f.keys())
        print(w_f['000000'])