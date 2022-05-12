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

from datasets import CLEVRDataset, build_predicates
from networks import EmbeddingNet, ReadoutNet
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml


def calc_f1(pred, target, mask, majority):
    pred[:, majority] = ~pred[:, majority]
    target[:, majority] = ~target[:, majority]
    tp = ((pred & target) * mask).sum(axis=0)
    fp = ((pred & ~target) * mask).sum(axis=0)
    fn = ((~pred & target) * mask).sum(axis=0)
    precision = tp / (tp + fp) * 100
    recall = tp / (tp + fn) * 100
    f1 = 2 * precision * recall / (precision + recall)
    f1[np.isnan(f1)] = 0
    return f1


def bar_plot_group(data, labels, keys, gap, width, legloc, ylabel, title, legend=True):
    n_bars = len(labels)
    n_groups = len(keys)
    x = np.arange(n_groups) * gap  # the label locations
    left = x - width * (n_bars - 1) / 2
    for i, d in enumerate(data):
        vals = [d[key] for key in keys]
        rects = plt.bar(left + i * width, vals, width, label=labels[i])
        autolabel(rects, plt.gca())
    keys = [f'on_{key}' if key in ['tabletop', 'bookshelf'] else key for key in keys]
    # plt.xticks(x, keys, rotation=60, fontsize=14)
    plt.xticks(x, keys, rotation=0, fontsize=20)
    # plt.yticks(np.arange(0, 101, 20), np.arange(0, 101, 20), fontsize=14)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(title, fontsize=18)
    if legend:
        plt.legend(loc=legloc, fontsize=14)


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom'
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_dir', default='data/geospa_depth_split/')
    parser.add_argument('--split', default='valA')
    parser.add_argument('--img_h', type=int, default=320)
    parser.add_argument('--img_w', type=int, default=480)
    parser.add_argument('--obj_h', type=int, default=32)
    parser.add_argument('--obj_w', type=int, default=32)
    parser.add_argument('--n_objects', type=int, default=10)
    parser.add_argument('--n_views', type=int, default=1)
    parser.add_argument('--multiview', action='store_true')
    parser.add_argument('--depth', action='store_true')
    parser.add_argument('--xyz', action='store_true')
    parser.add_argument('--world', action='store_true')
    parser.add_argument('--obj_depth', action='store_true')
    # Model
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--mean_pool', action='store_true')
    parser.add_argument('--type_emb_dim', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=512)
    # Evaluation
    parser.add_argument('--checkpoint', default='log/geospa_train_split_0428/epoch_40.pth')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_worker', type=int, default=1)
    parser.add_argument('--results_file', default='results_file.txt')
    args = parser.parse_args()

    objects = [f'object{i:02d}' for i in range(args.n_objects)]
    pred_cfg = {'unary': [], 'binary': ['%s front_of %s', '%s right_of %s', '%s contains %s', '%s supports %s']}
    predicates = build_predicates(objects, pred_cfg['unary'], pred_cfg['binary'])

    types = set()
    for pred in pred_cfg['unary'] + pred_cfg['binary']:
        pref = pred[3:-3]
        if pref == 'on_surface':
            types.add(f"on_{pred.split(', ')[-1][:-1]}")
        else:
            types.add(pref)

    # data = CLEVRDataset(
    #     f'{args.data_dir}/{args.split}.h5',
    #     f'{args.data_dir}/objects.h5',
    #     args.n_objects, rand_patch=False
    # )
    # loader = DataLoader(data, args.batch_size, num_workers=args.n_worker)
    #
    # model = EmbeddingNet(
    #     (args.img_w, args.img_h), args.patch_size, args.n_objects,
    #     args.width, args.layers, args.heads
    # )
    # head = ReadoutNet(args.width, args.hidden_dim, 0, 4)
    #
    # checkpoint = torch.load(args.checkpoint, map_location='cpu')
    # model.load_state_dict(checkpoint['model'])
    # head.load_state_dict(checkpoint['head'])
    # model = model.cuda().eval()
    # head = head.cuda().eval()
    #
    # predictions = []
    # targets = []
    # masks = []
    # for img, obj_patches, target, mask in tqdm(loader):
    #     img = img.cuda()
    #     obj_patches = obj_patches.cuda()
    #     with torch.no_grad():
    #         emb, attn = model(img, obj_patches)
    #         logits = head(emb)
    #     predictions.append((logits > 0).cpu().numpy())
    #     targets.append(target.bool().numpy())
    #     masks.append(mask.bool().numpy())
    # predictions = np.concatenate(predictions)
    # targets = np.concatenate(targets)
    # masks = np.concatenate(masks)
    #
    # print(predictions.shape, targets.shape, masks.shape)
    # np.save('predictions.npy', predictions)
    # np.save('targets.npy', targets)
    # np.save('masks.npy', masks)

    predictions = np.load('predictions.npy')
    targets = np.load('targets.npy')
    masks = np.load('masks.npy')
    print(predictions.shape, targets.shape, masks.shape)

    predicates_logit_indices = {'all': range(360), 'front_of': range(90), 'right_of': range(90, 180), 'contains': range(180, 270), 'supports': range(270, 360)}
    metrics = {}
    metrics['target_true'] = {}
    for predicate, logit_indices in predicates_logit_indices.items():
        metrics['target_true'][predicate] = np.sum(targets[:, logit_indices] * masks[:, logit_indices]) / np.sum(masks[:, logit_indices]) * 100
    metrics['prediction_true'] = {}
    for predicate, logit_indices in predicates_logit_indices.items():
        metrics['prediction_true'][predicate] = np.sum(predictions[:, logit_indices] * masks[:, logit_indices]) / np.sum(masks[:, logit_indices]) * 100
    metrics['predicate_accuracy'] = {}
    for predicate, logit_indices in predicates_logit_indices.items():
        metrics['predicate_accuracy'][predicate] = np.sum((predictions[:, logit_indices] == targets[:, logit_indices]) * masks[:, logit_indices]) / np.sum(masks[:, logit_indices]) * 100
    metrics['scene_accuracy'] = {}
    for predicate, logit_indices in predicates_logit_indices.items():
        a = np.sum((predictions[:, logit_indices] == targets[:, logit_indices]) * masks[:, logit_indices], axis=-1) / np.sum(masks[:, logit_indices], axis=-1)
        metrics['scene_accuracy'][predicate] = np.nansum(a) / np.sum(np.isreal(a)) * 100
    metrics['scene_all_accuracy'] = {}
    for predicate, logit_indices in predicates_logit_indices.items():
        a = np.all((predictions[:, logit_indices] == targets[:, logit_indices]) | ~masks[:, logit_indices], axis=-1)
        metrics['scene_all_accuracy'][predicate] = np.nansum(a) / np.sum(np.isreal(a)) * 100
    metrics['predicate_precision'] = {}
    metrics['predicate_recall'] = {}
    metrics['predicate_f1'] = {}
    for predicate, logit_indices in predicates_logit_indices.items():
        tp = ((predictions[:, logit_indices] & targets[:, logit_indices]) * masks[:, logit_indices]).sum()
        fp = ((predictions[:, logit_indices] & ~targets[:, logit_indices]) * masks[:, logit_indices]).sum()
        fn = ((~predictions[:, logit_indices] & targets[:, logit_indices]) * masks[:, logit_indices]).sum()
        precision = tp / (tp + fp) * 100
        recall = tp / (tp + fn) * 100
        f1 = 2 * precision * recall / (precision + recall)
        metrics['predicate_precision'][predicate] = precision
        metrics['predicate_recall'][predicate] = recall
        metrics['predicate_f1'][predicate] = f1
    print(metrics)

    json.dump(metrics, open(args.results_file, 'w'))

    metrics_to_graph = {'scene_accuracy': metrics['scene_accuracy'],
                 'scene_all_accuracy': metrics['scene_all_accuracy'],
                 'predicate_f1': metrics['predicate_f1'],
                 'target_true': metrics['target_true']}
    fig, axs = plt.subplots(len(metrics_to_graph), 1)
    for i, (metric_name, metric) in enumerate(metrics_to_graph.items()):
        width = 10
        xs = numpy.arange(len(metric.keys())) * width * 2
        heights = [value for predicate, value in metric.items()]
        predicates = [predicate for predicate, value in metric.items()]
        bars = axs[i].bar(xs, heights, width, label=predicates)
        for j in range(len(xs)):
            axs[i].annotate(
                f"{heights[j]:.1f}",
                xy=(xs[j], heights[j]),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom'
            )
        axs[i].set_ylim([0, 150])
        axs[i].set_xticks(numpy.arange(len(metric.keys())) * 2 * width, metric.keys())
        axs[i].set_title(metric_name)
    plt.subplots_adjust(hspace=0.75)
    plt.show()