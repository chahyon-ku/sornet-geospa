import h5py
import numpy
import torch
from networks import EmbeddingNet, ReadoutNet
import io
import tkinter
import os


relations = {'left': 0, 'right': 1, 'front': 2, 'behind': 3}
relation_phrases = {
    'left': 'to the left of',
    'right': 'to the right of',
    'front': 'in front of',
    'behind': 'behind'
}

objects_h5 = h5py.File('data/clevr_cogent/objects.h5')
scenes_h5 = h5py.File('data/clevr_cogent/valA.h5')

checkpoint_path = 'models/clevr_cogent.pth'
model = EmbeddingNet((320, 480), 32, 2, 768, 12, 12)
head = ReadoutNet(768, 512, 0, len(relations))
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model'])
head.load_state_dict(checkpoint['head'])
print('checkpoint loaded from', checkpoint_path)

