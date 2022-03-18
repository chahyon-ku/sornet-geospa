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
    parser.add_argument('--data_dir', default='data/clevr_cogent/')
    parser.add_argument('--split', default='valB')
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
    parser.add_argument('--checkpoint', default='models/clevr_cogent.pth')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_worker', type=int, default=2)
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

    writer = tensorboardX.SummaryWriter('log/clevr_test')

    correct = 0
    total = 0
    i = 0
    normalize_inverse = datasets.NormalizeInverse(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    relations = {'left': 0, 'right': 1, 'front': 2, 'behind': 3}
    relation_phrases = {
        'left': 'to the left of',
        'right': 'to the right of',
        'front': 'in front of',
        'behind': 'behind'
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
        correct += (pred[mask] == target[mask]).sum().item()
        total += mask.sum().item()

        index = 0
        obj1_index = 0
        obj2_index = 1
        relation = 0
        img_raw = datasets.denormalize_rgb(img[0].cpu())
        fig, (a0, a1, a2) = plt.subplots(
            1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [7, 2, 4]}
        )
        a0.imshow(img_raw)
        a0.set_title('Input image', fontsize=18)
        a0.axis('off')
        obj_img = numpy.ones((224, 96, 3)).astype('uint8') * 255
        obj_img[:96] = numpy.array(datasets.denormalize_rgb(obj_patches[index][obj1_index]).resize((96, 96)))
        obj_img[128:] = numpy.array(datasets.denormalize_rgb(obj_patches[index][obj2_index]).resize((96, 96)))
        a1.imshow(obj_img)
        a1.set_title('Query Object', fontsize=18)
        a1.axis('off')

        rel_phrase = relation_phrases['left']
        q_text = f"Is the {obj1_index} {rel_phrase}" \
                 f" the {obj2_index}?"
        q_text = q_text.split()
        q1 = ' '.join(q_text[:len(q_text) // 2])
        q2 = ' '.join(q_text[len(q_text) // 2:])
        a2.set_title('Question', fontsize=18)
        a2.text(0.5, 0.85, q1, fontsize=16, ha='center', va='center')
        a2.text(0.5, 0.75, q2, fontsize=16, ha='center', va='center')

        print(logits.shape)
        pred = logits[index].reshape(len(relations), -1)
        print(pred.shape)
        pred = pred[relation][0] > 0
        print(pred.shape)
        pred_text = 'Yes' if pred else 'No'
        a2.text(0.5, 0.5 * relation, 'Answer', fontsize=18, ha='center', va='center')
        a2.text(0.5, 0.2 + 0.5 * relation, f'SORNet: {pred_text}', fontsize=16, ha='center', va='center')
        a2.axis('off')
        plt.tight_layout()

        io_buffer = io.BytesIO()
        fig_size = fig.get_size_inches() * fig.dpi
        plt.show()
        print(fig_size)
        fig.savefig(io_buffer, format='raw', dpi=fig.dpi)
        io_buffer.seek(0)
        out_img = numpy.frombuffer(io_buffer.getvalue(), dtype=numpy.uint8)
        print(out_img.shape)
        out_img = numpy.reshape(out_img, (int(fig_size[0]), int(fig_size[1]), -1))
        writer.add_image('img'+str(i), out_img)
        print(obj_patches)
        i += 1
        break

    print('Total', total)
    print('Accuracy', correct / total * 100)
