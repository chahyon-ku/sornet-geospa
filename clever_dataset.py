import torch
import torchvision
import h5py
from PIL import Image
import io
import numpy


normalize_rgb = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

denormalize_rgb = torchvision.transforms.Compose([
    torchvision.transforms.NormalizeInverse(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    ),
    torchvision.transforms.ToPILImage(),
])


class CLEVRDataset(torch.utils.data.Dataset):
    def __init__(self, scene_file, obj_file, max_nobj, rand_patch):
        self.obj_file = obj_file
        self.obj_h5 = None
        self.scene_file = scene_file
        self.scene_h5 = None
        with h5py.File(scene_file, 'r') as scene_h5:
            self.scenes = list(scene_h5.keys())
        self.max_nobj = max_nobj
        self.rand_patch = rand_patch

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        if self.obj_h5 is None:
            self.obj_h5 = h5py.File(self.obj_file, 'r')
        if self.scene_h5 is None:
            self.scene_h5 = h5py.File(self.scene_file, 'r')

        scene = self.scene_h5[self.scenes[idx]]
        img = normalize_rgb(Image.open(io.BytesIO(scene['image'][()])).convert('RGB'))

        objects = scene['objects'][()].decode().split(',')
        obj_patches = []
        for obj in objects:
            patch_idx = 0
            if self.rand_patch:
                patch_idx = torch.randint(len(self.obj_h5[obj]), ()).item()
            patch = normalize_rgb(Image.open(io.BytesIO(self.obj_h5[obj][patch_idx])))
            obj_patches.append(patch)
        for _ in range(len(obj_patches), self.max_nobj):
            obj_patches.append(torch.zeros_like(obj_patches[0]))
        obj_patches = torch.stack(obj_patches)

        relations, mask = [], []
        ids = numpy.arange(self.max_nobj)
        for relation in scene['relations']:
            for k in range(1, self.max_nobj):
                for i, j in zip(ids, numpy.roll(ids, -k)):
                    if i >= len(objects) or j >= len(objects):
                        relations.append(0)
                        mask.append(0)
                    else:
                        relations.append(relation[i][j])
                        mask.append(relation[i][j] != -1)
        relations = torch.tensor(relations).float()
        mask = torch.tensor(mask).float()

        return img, obj_patches, relations, mask