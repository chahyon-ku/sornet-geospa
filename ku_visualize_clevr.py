import h5py
from PIL import Image
from datasets import normalize_rgb

scene_h5 = h5py.File('data/clevr_cogent/valA.h5', 'r')

for key in scene_h5.keys():
    print(key)

scene = scene_h5[list(scene_h5.keys())[0]]

relations, mask = [], []
for relation in scene['relations']:
    print(relation)