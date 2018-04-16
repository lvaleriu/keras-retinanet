import json
import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from ..preprocessing.generator import Generator
from ..utils.image import read_image_bgr


class LabelBoxGenerator(Generator):
    def __init__(
            self,
            subset, main_dir, json_filename,
            add_negative_examples=True,
            coco_data_dir='/media/work/image/coco',
            **kwargs
    ):
        self.set_name = subset
        self.add_negative_examples = add_negative_examples

        annotation_cache_json = 'labelbox_cache.json'

        if not os.path.exists(annotation_cache_json):
            # load raw data from labelbox exported json and save
            # generated annotations to a cache file.
            # we filter incoherent boxes: x1 = x2 or y1 = y2
            self.annotations = LabelBoxGenerator.__load_data__(main_dir, json_filename)
            json.dump(self.annotations, open(annotation_cache_json, "w"))
        else:
            with open(annotation_cache_json, 'r') as f:
                self.annotations = json.loads(f.read())
                ignored = []
                annotations = []

                for a in self.annotations:
                    if len(a['boxes']) > 0:
                        annotations.append(a)
                    else:
                        ignored.append(a)

                print ('accepted: {} annotations. total ignored: {} annotations'.format(len(annotations), len(ignored)))

                self.annotations = annotations

        train_annotations, test_annotations = train_test_split(self.annotations, test_size=0.25, random_state=0)
        self.annotations = train_annotations if subset == 'train' else test_annotations

        if add_negative_examples:
            from ..preprocessing.coco import CocoGenerator

            # add negative examples
            train_generator = CocoGenerator(
                coco_data_dir,
                '{}2017'.format(subset),
                batch_size=1
            )

            print('Adding {} negative examples'.format(len(self.annotations)))

            permutation = range(0, train_generator.size())
            random.shuffle(permutation)
            permutation = permutation[:len(self.annotations)]

            for image_index in permutation:
                image_info = train_generator.coco.loadImgs(train_generator.image_ids[image_index])[0]
                img_path = os.path.join(train_generator.data_dir, 'images', train_generator.set_name, image_info['file_name'])
                self.annotations.append({'path': img_path})

        print ('{}: {} accepted annotations'.format(subset, len(self.annotations)))

        super(LabelBoxGenerator, self).__init__(**kwargs)

    @staticmethod
    def __extract_bbox_data_v2__(boxes):
        img_boxes = []

        for boxes in boxes:
            box = []
            for coord in boxes:
                x = coord['x']
                y = coord['y']

                box.append((int(round(x)), int(round(y))))

            # x1, y1, x2, y2
            x1 = box[0][0]
            y1 = box[1][1]
            x2 = box[2][0]
            y2 = box[0][1]

            img_boxes.append((x1, y1, x2, y2))

        return img_boxes

    @staticmethod
    def __extract_bbox_data__(boxes):
        boxes = boxes.replace('MULTIPOLYGON ', '')
        boxes = boxes.replace('(((', '((')
        boxes = boxes.replace(')))', '')
        boxes = boxes.replace('((', '')

        boxes_data = boxes.split(')),')
        img_boxes = []

        for boxes in boxes_data:
            if len(boxes) > 0:
                bboxes = boxes.split(', ')

                box = []
                for coord in bboxes:
                    c = coord.strip().split(' ')
                    x = float(c[0])
                    y = float(c[1])

                    box.append((int(round(x)), int(round(y))))

                # x1, y1, x2, y2
                x1 = box[0][0]
                y1 = box[1][1]
                x2 = box[2][0]
                y2 = box[0][1]

                img_boxes.append((x1, y1, x2, y2))

        return img_boxes

    @staticmethod
    def __load_data__(main_dir, json_filename):
        json_path = os.path.join(main_dir, json_filename)
        all_boxes = []

        with open(json_path, mode='r') as f:
            data = json.load(f)

            for i, d in enumerate(tqdm(data)):
                url = d[u'Labeled Data']
                dlabel = d[u'Label']

                img_path = url.replace('http://cryptorecorder.ddns.net/', '')
                img_path = os.path.join(main_dir, img_path)

                # load image
                image = read_image_bgr(img_path)
                h, w = image.shape[:2]

                new_img_boxes = []

                if 'pylon' in dlabel:
                    img_boxes = LabelBoxGenerator.__extract_bbox_data_v2__(dlabel['pylon'])

                    for idx, t in enumerate(img_boxes):
                        (x1, y1, x2, y2) = t

                        y2 = h - y2
                        y1 = h - y1

                        if y2 < y1:
                            tmp = y1
                            y1 = y2
                            y2 = tmp

                        if x2 < x1:
                            tmp = x1
                            x1 = x2
                            x2 = tmp

                        if x2 > w:
                            x2 = w
                        if y2 > h:
                            y2 = h

                        if x1 < 0:
                            x1 = 0
                        if x2 < 0:
                            x2 = 0

                        if y1 < 0:
                            y1 = 0
                        if y2 < 0:
                            y2 = 0

                        if x2 == x1 or y2 == y1:
                            print ('x1: {} x2: {} y1: {} y2: {} user: {}'.format(x1, x2, y1, y2, d[u'Created By']))
                            continue

                        new_img_boxes.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

                all_boxes.append({'h': h, 'w': w, 'path': img_path, 'boxes': new_img_boxes})

        return all_boxes

    def size(self):
        return len(self.annotations)

    def num_classes(self):
        return 1

    def name_to_label(self, name):
        raise NotImplementedError()

    def label_to_name(self, label):
        return 'pylon'

    def image_aspect_ratio(self, image_index):
        # PIL is fast for metadata
        image = Image.open(self.annotations[image_index]['path'])
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        return read_image_bgr(self.annotations[image_index]['path'])

    def load_annotations(self, image_index):
        annotations = self.annotations[image_index]

        if self.add_negative_examples and 'boxes' not in annotations:
            return np.zeros((0, 5))

        boxes = np.zeros((len(annotations['boxes']), 5))

        for idx, ann in enumerate(annotations['boxes']):
            cls_id = 0

            boxes[idx, 0] = ann['x1']
            boxes[idx, 1] = ann['y1']
            boxes[idx, 2] = ann['x2']
            boxes[idx, 3] = ann['y2']
            boxes[idx, 4] = cls_id

        return boxes
