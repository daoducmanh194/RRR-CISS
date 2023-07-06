import os
import glob
import sys
import json
import numpy as np

import torch
import torch.utils.data as data
from PIL import Image

from .tasks import get_dataset_list, get_tasks


# Converting the id to the train_id. Many objects have a train id at
# 255 (unknown / ignored).
# See there for more information:
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
id_to_trainid = {
    0: 0, # unlabelled + background
    1: 255,
    2: 255,
    3: 255,
    4: 255,
    5: 255,
    6: 255,
    7: 1,   # road
    8: 2,   # sidewalk
    9: 255,
    10: 255,
    11: 3,  # building
    12: 4,  # wall
    13: 5,  # fence
    14: 255,
    15: 255,
    16: 255,
    17: 6,  # pole
    18: 255,
    19: 7,  # traffic light
    20: 8,  # traffic sign
    21: 9,  # vegetation
    22: 10,  # terrain
    23: 11, # sky
    24: 12, # person
    25: 13, # rider
    26: 14, # car
    27: 15, # truck
    28: 16, # bus
    29: 255,
    30: 255,
    31: 17, # train
    32: 18, # motorcycle
    33: 19, # bicycle
    -1: 255
}


def filter_images(dataset, labels, labels_old=None, overlap=True):
    # Filter images without any label in LABELS (using labels not reordered)
    idxs = []

    if 0 in labels:
        labels.remove(0)

    print(f"Filtering images...")
    if labels_old is None:
        labels_old = []
    labels_cum = labels + labels_old + [0,255]
    if overlap:
        fil = lambda c: any(x in labels for x in c)
    else:
        fil = lambda c: any(x in labels for x in c) and all(x in labels_cum for x in c)

    for i in range(len(dataset)):
        tgt = np.unique(np.array(dataset[i][1],dtype=np.int64).flatten())
        cls = [id_to_trainid.get(x,255) for x in tgt]
        if fil(cls):
            idxs.append(i)
        if i % 500 == 0:
            print(f"\t{i}/{len(dataset)} ...")
    print('no of images in current task : ', len(idxs))
    return idxs


class CityScapesSegmentation(data.Dataset):
    def __init__(self, args, image_set='train', transform=None, cil_step=0, mem_size=0):
        self.root = args.data_root
        self.task = args.task
        self.overlap = args.overlap
        self.unknown = args.unknown

        self.image_set = image_set
        self.transform = transform

        cityscapes_root = '/datasets/data/cityscapes'
        image_folder = os.path.join(self.root, 'leftImg8bit')
        annotation_folder = os.path.join(self.root, 'gtFine')
        salmap_folder = os.path.join(self.root, 'saliency_map_leftImg8bit_picanet')

        if not os.path.isdir(self.root):
            raise RuntimeError('Dataset not found or corrupted')
        
        assert os.path.exists(annotation_folder), "Annotations folder not found."

        self.target_cls = get_tasks('cityscapes', self.task, cil_step)
        self.target_cls += [255]

        if image_set == 'train':
            self.images = [  # Add 18 train cities
                (
                    path,
                    os.path.join(
                        annotation_folder,
                        "train",
                        path.split("/")[-2],
                        path.split("/")[-1][:-15] + "gtFine_labelIds.png"
                    )
                ) for path in sorted(glob.glob(os.path.join(image_folder, "train/*/*.png")))
            ]
            print('images ', len(self.images))
        elif image_set == 'val':
            self.images = [  # Add 3 validation cities
                (
                    path,
                    os.path.join(
                        annotation_folder,
                        "val",
                        path.split("/")[-2],
                        path.split("/")[-1][:-15] + "gtFine_labelIds.png"
                    )
                ) for path in sorted(glob.glob(os.path.join(image_folder, "val/*/*.png")))
            ]
        elif image_set == 'test':
            self.images = [
                (
                    path,
                    os.path.join(
                        annotation_folder,
                        "test",
                        path.split("/")[-2],
                        path.split("/")[-1][:-15] + "gtFine_labelIds.png"
                    )
                ) for path in sorted(glob.glob(os.path.join(image_folder, "test/*/*.png")))
            ]
        elif image_set == 'memory':
            for s in range(cil_step):
                self.target_cls += get_tasks('cityscapes', self.task, s)
            
            memory_json = os.path.join(cityscapes_root, 'memory.json')

            with open(memory_json, "r") as json_file:
                memory_list = json.load(json_file)

            file_names = memory_list[f"step_{cil_step}"]["memory_list"]
            print("... memory list : ", len(file_names), self.target_cls)
            
            while len(file_names) < args.batch_size:
                file_names = file_names * 2

        self.transform = transform

        # class re-ordering
        all_steps = get_tasks('cityscapes', self.task)
        all_classes = []
        for i in range(len(all_steps)):
            all_classes += all_steps[i]
            
        self.ordering_map = np.zeros(256, dtype=np.uint8) + 255
        self.ordering_map[:len(all_classes)] = [all_classes.index(x) for x in range(len(all_classes))]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        try:
            img = Image.open(self.images[index][0]).convert('RGB')
            target = Image.open(self.images[index][1])
            sal_map = Image.fromarray(np.ones(target.size[::-1], dtype=np.uint8))

            # re-define target label according to the CIL case
            target = self.gt_label_mapping(target)
        except Exception as e:
            raise Exception(f"Index: {index}, len: {len(self)}, message: {str(e)}")

        if self.transform is not None:
            img, target, sal_map = self.transform(img, target, sal_map)
        
        # add unknown label, background index: 0 -> 1, unknown index: 0
        if self.image_set == 'train' and self.unknown:
            
            target = torch.where(target == 255, 
                                 torch.zeros_like(target) + 255,  # keep 255 (uint8)
                                 target+1) # unknown label
            
            unknown_area = (target == 1)
            target = torch.where(unknown_area, torch.zeros_like(target), target)

        # print("img: ", img)
        # print("target: ", target.long)
        # print("sal_map: ", sal_map)
        # print("file_name: ", None)
        return img, target.long(), sal_map, {}

    def __len__(self):
        return len(self.images)
    
    def gt_label_mapping(self, gt):
        gt = np.array(gt, dtype=np.uint8)
        if self.image_set != 'test':
            gt = np.where(np.isin(gt, self.target_cls), gt, 0)
        gt = self.ordering_map[gt]
        gt = Image.fromarray(gt)

        return gt
    
    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]
