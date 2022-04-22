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
import io

import numpy
import torchvision
from PIL import Image
from matplotlib import pyplot as plt

import datasets
from datasets import CLEVRDataset
from networks import EmbeddingNet, ReadoutNet
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import torch
import tensorboardX
import time


def log(
        writer, global_step, split, epoch, idx, total,
        batch_time, data_time, avg_loss, avg_acc, pred_types=None
    ):
    print(
        f'Epoch {(epoch+1):02d} {split.capitalize()} {idx:04d}/{total:04d} '
        f'Batch time {batch_time:.3f} Data time {data_time:.3f} '
        f'Loss {avg_loss.item():.4f} Accuracy {avg_acc.mean().item():.2f}'
    )
    writer.add_scalar(f'{split}/loss', avg_loss, global_step)
    writer.add_scalar(f'{split}/accuracy', avg_acc.mean().item(), global_step)
    for a, name in zip(avg_acc, pred_types.keys()):
        writer.add_scalar(f'{split}/accuracy_{name}', a.item(), global_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    # parser.add_argument('--data_dir', default='data/geospa/')
    # parser.add_argument('--split', default='sample_2')
    parser.add_argument('--data_dir', default='data/geospa/')
    parser.add_argument('--split', default='train')
    parser.add_argument('--max_nobj', type=int, default=10)
    parser.add_argument('--img_h', type=int, default=320)
    parser.add_argument('--img_w', type=int, default=480)
    # Model
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--d_hidden', type=int, default=512)
    parser.add_argument('--n_relation', type=int, default=4)
    # Evaluation
    #parser.add_argument('--checkpoint', default='log/log_geospa/epoch_40.pth')
    parser.add_argument('--checkpoint', default='log/log_geospa/epoch_40.pth')
    #parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    #parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_worker', type=int, default=0)
    args = parser.parse_args()

    data = CLEVRDataset(
        f'{args.data_dir}/{args.split}.h5',
        f'{args.data_dir}/objects.h5',
        args.max_nobj, rand_patch=False
    )
    loader = DataLoader(data, args.batch_size, num_workers=args.n_worker)

    model = EmbeddingNet(
        (args.img_w, args.img_h), args.patch_size, args.max_nobj,
        args.width, args.layers, args.heads
    )
    head = ReadoutNet(args.width, args.d_hidden, 0, args.n_relation)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    head.load_state_dict(checkpoint['head'])
    model = model.cuda().eval()
    head = head.cuda().eval()
    # for name, parameter in model.named_parameters():
    #     print(name, parameter.shape)

    writer = tensorboardX.SummaryWriter('log/geospa_test3/' + str(int(time.time())))

    correct = 0
    total = 0
    batch_i = 0
    normalize_inverse = datasets.NormalizeInverse(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    # relations = ['left', 'right', 'front', 'behind']
    # relation_phrases = {
    #     'left': 'left of',
    #     'right': 'right of',
    #     'front': 'in front of',
    #     'behind': 'behind'
    # }
    relations = ['front', 'right', 'contain', 'support']
    relation_phrases = {
        'front': 'in front of',
        'right': 'right of',
        'contain': 'contains',
        'support': 'supports'
    }
    for img, obj_patches, target, mask in tqdm(loader):
        img = img.cuda()
        obj_patches = obj_patches.cuda()
        with torch.no_grad():
            emb, attn = model(img, obj_patches)
            logits = head(emb)
            pred = (logits > 0).int().cpu()
        target = target.int()

        mask = mask.bool()
        max_obj_i = numpy.zeros(obj_patches.shape[0], dtype=int)
        for img_i in range(obj_patches.shape[0]):
            for obj_i in range(10):
                if (numpy.array(obj_patches[img_i, obj_i].cpu()).swapaxes(-1, -3) != numpy.array([0, 0, 0])).any():
                    max_obj_i[img_i] = obj_i

        correct += (pred[mask] == target[mask]).sum().item()
        total += mask.sum().item()

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
            obj_img[32 * (2 * i + 1):32 * (2 * i + 2), :32] = numpy.array(datasets.denormalize_rgb(obj_patches[index][2 * i + 1]))
        a1.imshow(obj_img)
        a1.set_title('Query Object', fontsize=18)
        a1.axis('off')

        target = target[index].reshape(len(relations), -1)
        pred = logits[index].reshape(len(relations), -1)
        mask = mask[index].reshape(len(relations), -1)
        row_count = 0
        # print('len(objects)', max_obj_i[0] + 1)
        # for rel_i in range(4):
        #     print(numpy.array(target[rel_i]))
        for obj1_i in range(args.max_nobj):
            for k in range(1, args.max_nobj):
                obj2_i = (obj1_i + k) % args.max_nobj
                for rel_i in range(4):
                    rel_mask = mask[rel_i][(k - 1) * args.max_nobj + obj1_i] > 0
                    rel_pred = pred[rel_i][(k - 1) * args.max_nobj + obj1_i] > 0
                    rel_true = target[rel_i][(k - 1) * args.max_nobj + obj1_i] > 0
                    if not rel_mask or (not rel_pred and not rel_true):
                        continue

                    rel = relations[rel_i]
                    rel_phrase = relation_phrases[rel]
                    pred_text = '' if rel_pred else 'not '
                    pred_text = pred_text + rel_phrase
                    color = (0, 0, 0)
                    if rel_pred and not rel_true: # false positive
                        color = (1, 0, 0)
                    elif not rel_pred and rel_true: # false negative
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

        #io_buffer = io.BytesIO()
        fig_size = fig.get_size_inches() * fig.dpi
        #fig.savefig(io_buffer, format='raw', dpi=fig.dpi)
        fig.savefig('./geospa_test_images/' + str(batch_i) + '.png', format='png', dpi=fig.dpi)

        #io_buffer.seek(0)
        #out_img = numpy.frombuffer(io_buffer.getvalue(), dtype=numpy.uint8)
        #out_img = numpy.reshape(out_img, (int(fig_size[1]), int(fig_size[0]), -1))
        #writer.add_image('img'+str(batch_i), out_img, dataformats='HWC')
        print('wrote', 'img'+str(batch_i))
        batch_i += 1
        if batch_i == 100:
            break

    print('Total', total)
    print('Accuracy', correct / total * 100)
