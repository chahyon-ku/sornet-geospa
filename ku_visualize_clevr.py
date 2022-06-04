'''
MIT License

Copyright (c) 2022 Wentao Yuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import argparse
import io
import os

import h5py
import json
import torch
from PIL import Image
from datasets import normalize_rgb, build_predicates
from io import BytesIO
from matplotlib import pyplot as plt
from networks import EmbeddingNet, ReadoutNet
from train_leonardo import plot
import cv2
import numpy


file_paths = ['./data/geospa_half3/val_default.h5',
              './data/geospa_half3/val_color.h5',
              './data/geospa_half3/val_size.h5',
              './data/geospa_half3/val_shape.h5',
              './data/geospa_half3/val_material.h5']
save_paths = ['./geospa_half3/val_default_scenes/',
              './geospa_half3/val_color_scenes/',
              './geospa_half3/val_size_scenes/',
              './geospa_half3/val_shape_scenes/',
              './geospa_half3/val_default_scenes/']

# for file_path, save_path in zip(file_paths, save_paths):
#     with h5py.File(file_path, 'r') as f:
#         # print(f.keys())
#         # print(f['000000'].keys())
#         # print(f['000000']['image'])
#         # print(f['000000']['relations'])
#         os.makedirs(save_path, exist_ok=True)
#         for i in range(100):
#             scene_key = '%06d' % i
#             print(scene_key)
#             img_pil = Image.open(BytesIO(numpy.array(f[scene_key]['image'][()])))
#             img_np = numpy.array(img_pil)
#             img_np[:, :, -1] = 255
#             #img_np = img_np[:,:,-1]
#             img_pil = Image.fromarray(img_np)
#             #img_pil = img_pil.convert('RGB')
#             img_pil.save(save_path + scene_key + '.png', 'png')
        # print(f['CLEVR_new_000000']['objects'][0])
        # print(f['CLEVR_new_000000']['relations'].keys())
        # print(f['CLEVR_new_000000']['relations']['contain'][()])

        # img_np = numpy.array(img_pil.convert('RGB'), dtype=numpy.uint8)
        # img_np = normalize_rgb(Image.open(BytesIO(f['000000']['image'][()])).convert('RGB'))
        # cv2.imshow('img', img_np)
        # cv2.waitKey()

with h5py.File('../Summer_research/output/scenes.h5', 'r') as f:
    print(f.keys())
    print(f['000000'].keys())
    print(f['000000']['images'].keys())
    print(f['000000']['images']['0'])
    print(f['000000']['relations'])
    print(f['000000']['relations'].keys())
    for key in f:
        img_pil = Image.open(BytesIO(f[key]['images']['1'][()])).convert('RGB')
        img_np = numpy.array(img_pil, dtype=numpy.uint8)
        cv2.imshow('img', img_np)
        cv2.waitKey()

#
# with h5py.File('data/geospa_depth/objects.h5', 'r') as f:
#     print(f.keys())
#     for key in f:
#         print(f[key])
#         img_pil = Image.open(BytesIO(f[key][0]))
#         img_np = numpy.array(img_pil, dtype=numpy.uint8)[:, :, :3]
#         print(img_np)
#         cv2.imshow('img', numpy.array(img_np, dtype=numpy.uint8))
#         cv2.waitKey()