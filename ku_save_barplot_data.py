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
import os

import numpy

from datasets import CLEVRMultiviewDataset, build_predicates
from networks import EmbeddingNetMultiview, ReadoutNet
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
import torch
import numpy as np



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_dir', default='data/geospa_2view/')
    parser.add_argument('--split', default='train')
    parser.add_argument('--img_h', type=int, default=320)
    parser.add_argument('--img_w', type=int, default=480)
    parser.add_argument('--obj_h', type=int, default=32)
    parser.add_argument('--obj_w', type=int, default=32)
    parser.add_argument('--n_objects', type=int, default=10)
    parser.add_argument('--n_views', type=int, default=1)
    # Model
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--mean_pool', action='store_true')
    parser.add_argument('--type_emb_dim', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=512)
    # Evaluation
    parser.add_argument('--checkpoint', default='log/geospa_1view/epoch_80.pth')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_worker', type=int, default=1)
    parser.add_argument('--results_dir', default='./geospa_1view/train_bar/')
    args = parser.parse_args()

    objects = [f'object{i:02d}' for i in range(args.n_objects)]
    #pred_cfg = {'unary': [], 'binary': ['%s front_of %s', '%s right_of %s', '%s contains %s', '%s supports %s']}
    pred_cfg = {'unary': [], 'binary': ['%s left_of %s', '%s right_of %s', '%s front_of %s', '%s behind_of %s',
                                        '%s can_contain %s', '%s can_support %s', '%s supports %s', '%s contains %s']}
    predicates = build_predicates(objects, pred_cfg['unary'], pred_cfg['binary'])

    types = set()
    for pred in pred_cfg['unary'] + pred_cfg['binary']:
        pref = pred[3:-3]
        if pref == 'on_surface':
            types.add(f"on_{pred.split(', ')[-1][:-1]}")
        else:
            types.add(pref)

    data = CLEVRMultiviewDataset(
        f'{args.data_dir}/{args.split}.h5',
        f'{args.data_dir}/objects.h5',
        args.n_objects, rand_patch=False
    )
    loader = DataLoader(data, args.batch_size, num_workers=args.n_worker)

    model = EmbeddingNetMultiview(
        (args.img_w, args.img_h), args.patch_size, args.n_objects,
        args.width, args.layers, args.heads, 1, [3]
    )
    head = ReadoutNet(args.width, args.hidden_dim, len(pred_cfg['unary']), len(pred_cfg['binary']))

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    head.load_state_dict(checkpoint['head'])
    model = model.cuda().eval()
    head = head.cuda().eval()

    predictions = []
    targets = []
    masks = []
    for imgs, obj_patches, target, mask in tqdm(loader):
        imgs = [img.cuda() for img in imgs]
        obj_patches = obj_patches.cuda()
        with torch.no_grad():
            emb, attn = model(imgs, obj_patches)
            logits = head(emb)
        predictions.append((logits > 0).cpu().numpy())
        targets.append(target.bool().numpy())
        masks.append(mask.bool().numpy())
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    masks = np.concatenate(masks)

    print(predictions.shape, targets.shape, masks.shape)
    os.makedirs(args.results_dir, exist_ok=True)
    np.save(args.results_dir + 'predictions.npy', predictions)
    np.save(args.results_dir + 'targets.npy', targets)
    np.save(args.results_dir + 'masks.npy', masks)
