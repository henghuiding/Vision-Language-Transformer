# encoding=utf8

import numpy as np
import os
from refer import REFER
import cv2
import argparse
from tqdm import tqdm
import json
parser = argparse.ArgumentParser(description='Data preparation')
parser.add_argument('--data_root',  type=str)
parser.add_argument('--output_dir',  type=str)
parser.add_argument('--dataset', type=str, choices=['refcoco', 'refcoco+', 'refcocog', 'refclef'], default='refcoco')
parser.add_argument('--split',  type=str, default='umd')
parser.add_argument('--generate_mask',  action='store_true')
args = parser.parse_args()
img_path = os.path.join(args.data_root, 'images', 'train2014')

h, w = (416, 416)

refer = REFER(args.data_root, args.dataset, args.split)

print('dataset [%s_%s] contains: ' % (args.dataset, args.split))
ref_ids = refer.getRefIds()
image_ids = refer.getImgIds()
print('%s expressions for %s refs in %s images.' % (len(refer.Sents), len(ref_ids), len(image_ids)))

print('\nAmong them:')
if args.dataset == 'refclef':
    if args.split == 'unc':
        splits = ['train', 'val', 'testA', 'testB', 'testC']
    else:
        splits = ['train', 'val', 'test']
elif args.dataset == 'refcoco':
    splits = ['train', 'val', 'testA', 'testB']
elif args.dataset == 'refcoco+':
    splits = ['train', 'val',  'testA', 'testB']
elif args.dataset == 'refcocog':
    splits = ['train', 'val', 'test']  # we don't have test split for refcocog right now.


for split in splits:
    ref_ids = refer.getRefIds(split=split)
    print('%s refs are in split [%s].' % (len(ref_ids), split))


def cat_process(cat):
    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11
    return cat


def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))


def prepare_dataset(dataset, splits, output_dir, generate_mask=False):
    ann_path = os.path.join(output_dir, 'anns', dataset)
    mask_path = os.path.join(output_dir, 'masks', dataset)
    if not os.path.exists(ann_path):
        os.makedirs(ann_path)
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)

    for split in splits:
        dataset_array = []
        ref_ids = refer.getRefIds(split=split)
        print('Processing split:{} - Len: {}'.format(split, np.alen(ref_ids)))
        for i in tqdm(ref_ids):
            ref_dict = {}

            refs = refer.Refs[i]
            bboxs = refer.getRefBox(i)
            sentences = refs['sentences']
            image_urls = refer.loadImgs(image_ids=refs['image_id'])[0]
            cat = cat_process(refs['category_id'])
            image_urls = image_urls['file_name']
            if dataset == 'refclef' and image_urls in ['19579.jpg', '17975.jpg', '19575.jpg']:
                continue
            box_info = bbox_process(bboxs)

            ref_dict['bbox'] = box_info
            ref_dict['cat'] = cat
            ref_dict['segment_id'] = i
            ref_dict['img_name'] = image_urls

            if generate_mask:
                cv2.imwrite(os.path.join(mask_path, str(i)+'.png'), refer.getMask(refs)['mask'] * 255)

            sent_dict = []
            for i, sent in enumerate(sentences):
                sent_dict.append({
                    'idx': i,
                    'sent_id': sent['sent_id'],
                    'sent': sent['sent'].strip()})

            ref_dict['sentences'] = sent_dict
            ref_dict['sentences_num'] = len(sent_dict)

            dataset_array.append(ref_dict)
        print('Dumping json file...')
        with open(os.path.join(output_dir, 'anns', dataset, split + '.json'), 'w') as f:
            json.dump(dataset_array, f)


prepare_dataset(args.dataset, splits, args.output_dir, args.generate_mask)
