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
import numpy
from matplotlib import pyplot as plt

import datasets
from datasets import LeonardoDataset, build_predicates
from networks import EmbeddingNet, ReadoutNet
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
import torch

unary_pred = [
    'on_surface(%s, left)', 'on_surface(%s, right)', 'on_surface(%s, far)',
    'on_surface(%s, center)', 'has_obj(robot, %s)', 'top_is_clear(%s)',
    'in_approach_region(robot, %s)'
]
binary_pred = ['stacked(%s, %s)', 'aligned_with(%s, %s)']


def calc_accuracy(pred, target):
    return (pred == target).sum(0) / target.shape[0] * 100


def calc_accuracy_allmatch(pred, target, keys, names):
    acc = {}
    acc['all'] = ((pred != target).sum(axis=1) == 0).sum() / pred.shape[0] * 100
    for key in keys:
        mask = [key in name for name in names]
        if sum(mask) > 0:
            correct = ((pred[:, mask] != target[:, mask]).sum(axis=1) == 0).sum()
            acc[key] = correct / pred.shape[0] * 100
        else:
            acc[key] = 0
    return acc


def calc_f1(pred, target):
    majority_is_one = target.shape[0] - target.sum(axis=0) < target.sum(axis=0)
    pred[:, majority_is_one] = ~pred[:, majority_is_one]
    target[:, majority_is_one] = ~target[:, majority_is_one]
    tp = (pred & target).sum(axis=0)
    fp = (pred & ~target).sum(axis=0)
    fn = (~pred & target).sum(axis=0)
    precision = tp / (tp + fp) * 100
    recall = tp / (tp + fn) * 100
    f1 = 2 * precision * recall / (precision + recall)
    f1[np.isnan(f1)] = 0
    return f1


def split_avg(data, keys, names):
    avg = {'all': np.mean(data)}
    for key in keys:
        mask = [key in name for name in names]
        if sum(mask) > 0:
            avg[key] = np.mean(data[mask])
        else:
            avg[key] = 0
    return avg


def create_and_write_image(img, obj_patches, gripper, target):
    #mask = numpy.array(mask.bool().cpu(), dtype=bool)
    max_obj_i = numpy.zeros(obj_patches.shape[0], dtype=int)
    for img_i in range(obj_patches.shape[0]):
        for obj_i in range(10):
            if (numpy.array(obj_patches[img_i, obj_i].cpu()).swapaxes(-1, -3) != numpy.array([0, 0, 0])).any():
                max_obj_i[img_i] = obj_i

    index = 0
    img_raw = datasets.denormalize_rgb(img[index].cpu())
    fig, (a0, a1, a2) = plt.subplots(
        1, 3, figsize=(15, 10), gridspec_kw={'width_ratios': [7, 2, 4]}
    )
    a0.imshow(img_raw)
    a0.set_title('Input image', fontsize=18)
    a0.axis('off')

    obj_img = numpy.ones((320, 32, 3)).astype('uint8') * 255
    for i in range(5):
        obj_img[32 * (2 * i):32 * (2 * i + 1), :32] = numpy.array(datasets.denormalize_rgb(obj_patches[index][2 * i]))
        obj_img[32 * (2 * i + 1):32 * (2 * i + 2), :32] = numpy.array(
            datasets.denormalize_rgb(obj_patches[index][2 * i + 1]))
    a1.imshow(obj_img)
    a1.set_title('Query Object', fontsize=18)
    a1.axis('off')

    target = target[index].reshape(len(unary_pred) + len(binary_pred), -1)
    pred = logits[index].reshape(len(unary_pred) + len(binary_pred), -1)
    #mask = mask[index].reshape(len(relations), -1)
    row_count = 0
    pair_count = -1
    for obj1_i in range(max_obj_i[0]):
        for obj2_i in range(max_obj_i[0]):
            if obj1_i == obj2_i:
                continue
            pair_count += 1
            # if (obj_img[32 * obj1_i:32 * (obj1_i + 1)] == numpy.array([122, 116, 104])).all()\
            #         or (obj_img[32 * obj2_i:32 * (obj2_i + 1)] == numpy.array([122, 116, 104])).all():
            #     continue
            for rel_i in range(4):
                #rel_mask = mask[rel_i][pair_count] > 0
                rel_pred = pred[rel_i][pair_count] > 0
                rel_true = target[rel_i][pair_count] > 0
                #if not rel_mask or (not rel_pred and not rel_true):
                #    continue

                rel = relations[rel_i]
                rel_phrase = relation_phrases[rel]
                pred_text = '' if rel_pred else 'not '
                pred_text = pred_text + rel_phrase
                color = (0, 0, 0)
                if rel_pred and not rel_true:  # false positive
                    color = (1, 0, 0)
                elif not rel_pred and rel_true:  # false negative
                    color = (0, 0, 1)
                a2.text(0.5, 1 - row_count * 0.025, pred_text, color=color, fontsize=12, ha='center', va='center')
                obj1_axis = a2.inset_axes([0.2, 1 - row_count * 0.025 - 0.0125, 0.1, 0.025])
                obj1_axis.imshow(obj_img[32 * obj1_i:32 * (obj1_i + 1)])
                obj1_axis.axis('off')
                obj2_axis = a2.inset_axes([0.7, 1 - row_count * 0.025 - 0.0125, 0.1, 0.025])
                obj2_axis.imshow(obj_img[32 * obj2_i:32 * (obj2_i + 1)])
                obj2_axis.axis('off')

                row_count += 1
    a2.axis('off')
    plt.tight_layout()

    io_buffer = io.BytesIO()
    fig_size = fig.get_size_inches() * fig.dpi
    fig.savefig(io_buffer, format='raw', dpi=fig.dpi)

    io_buffer.seek(0)
    out_img = numpy.frombuffer(io_buffer.getvalue(), dtype=numpy.uint8)
    out_img = numpy.reshape(out_img, (int(fig_size[1]), int(fig_size[0]), -1))
    writer.add_image('img' + str(batch_i), out_img, dataformats='HWC')
    batch_i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_dir')
    parser.add_argument('--split')
    parser.add_argument('--obj_file')
    parser.add_argument('--img_h', type=int, default=224)
    parser.add_argument('--img_w', type=int, default=224)
    parser.add_argument('--n_views', type=int, default=1)
    parser.add_argument('--n_objects', type=int, default=4)
    parser.add_argument('--objects', nargs='+')
    parser.add_argument('--colors', nargs='+')
    # Model
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--gripper', action='store_true')
    parser.add_argument('--d_hidden', type=int, default=512)
    # Evaluation
    parser.add_argument('--checkpoint')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_worker', type=int, default=0)
    args = parser.parse_args()

    if args.objects is None:
        objects = [f'object{i:02d}' for i in range(args.n_objects)]
    else:
        objects = args.objects
    pred_names = build_predicates(objects, unary_pred, binary_pred)

    loaders = []
    for v in range(args.n_views):
        data = LeonardoDataset(
            args.data_dir, args.split, pred_names, args.obj_file, args.colors,
            randpatch=False, view=v, randview=False, gripper=args.gripper
        )
        loaders.append(DataLoader(
            data, args.batch_size, pin_memory=True, num_workers=args.n_worker
        ))

    model = EmbeddingNet(
        (args.img_w, args.img_h), args.patch_size, len(objects),
        args.width, args.layers, args.heads
    )
    out_dim = args.width + 1 if args.gripper else args.width
    head = ReadoutNet(out_dim, args.d_hidden, len(unary_pred), len(binary_pred))
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    head.load_state_dict(checkpoint['head'])
    model = model.cuda().eval()
    head = head.cuda().eval()

    predictions = []
    targets = []
    loaders.insert(0, tqdm(range(len(loaders[0]))))
    for data in zip(*loaders):
        data = data[1:]
        batch_size = data[0][0].shape[0]
        logits = 0
        for img, obj_patches, gripper, target in data:
            with torch.no_grad():
                img = img.cuda()
                obj_patches = obj_patches.cuda()
                emb, attn = model(img, obj_patches)
                if args.gripper:
                    gripper = gripper.cuda()
                    emb = torch.cat([
                        emb, gripper[:, None, None].expand(-1, len(objects), -1)
                    ], dim=-1)
                logits += head(emb)

        predictions.append((logits > 0).cpu().numpy())
        targets.append(target.bool().numpy())
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    prefixes = [
        'on_surface', 'has_obj', 'top_is_clear',
        'in_approach_region', 'stacked', 'aligned_with'
    ]
    accuracy = split_avg(calc_accuracy(predictions, targets), prefixes, pred_names)
    accuracy_all = calc_accuracy_allmatch(predictions, targets, prefixes, pred_names)
    f1 = split_avg(calc_f1(predictions, targets), prefixes, pred_names)
    print('Accuracy')
    for key in accuracy:
        print(key, accuracy[key])
    print()
    print('All match accuracy')
    for key in accuracy_all:
        print(key, accuracy_all[key])
    print()
    print('F1 score')
    for key in f1:
        print(key, f1[key])
