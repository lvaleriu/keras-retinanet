import os
import random
import sys
import uuid

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    __package__ = "keras_retinanet.bin"

from keras_retinanet.preprocessing.open_images import OpenImagesGenerator


class StatsOpenImageGenerator(OpenImagesGenerator):
    def name_to_label(self, name):
        pass

    def stats(self):
        stats_per_label = dict([(self.id_to_labels[i], 0) for i in range(len(self.id_to_labels))])
        for f in self.annotations.iteritems():
            for ann in f[1]['boxes']:
                stats_per_label[self.id_to_labels[ann['cls_id']]] += 1

        print ('Stats per label: {}'.format(stats_per_label))

        for l, cnt in stats_per_label.iteritems():
            if cnt == 0:
                self.id_to_labels = dict([(i, lb) for i, lb in self.id_to_labels.iteritems() if lb != l])

        # print ('Id to labels: {}'.format(self.id_to_labels))


class CocoOpenImagesGenerator(StatsOpenImageGenerator):
    def __init__(self, **kwargs):
        labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
                           7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                           12: 'parking meter',
                           13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                           20: 'elephant',
                           21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
                           27: 'tie',
                           28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
                           34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
                           38: 'tennis racket',
                           39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
                           46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
                           52: 'hot dog',
                           53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
                           59: 'bed',
                           60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                           66: 'keyboard',
                           67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
                           73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
                           79: 'toothbrush'}

        labels_to_names = {0: 'person', 1: 'car'}

        kwargs['labels_filter'] = None
        self.use_all_classes = True

        super(CocoOpenImagesGenerator, self).__init__(**kwargs)

        self.id_to_labels, self.annotations = self.__filter_data__(labels_to_names.values())
        self.id_to_image_id = dict([(i, k) for i, k in enumerate(self.annotations)])

        self.group_images()

    def __filter_data__(self, coco_labels_filter):
        coco_labels_to_id = dict([(l, i) for i, l in enumerate(coco_labels_filter)])

        oid_labels = [str.lower(oid_label) for oid_label in self.id_to_labels.values()]
        oid_labels = sorted(oid_labels)

        oid_sub_labels_to_coco_id = {}

        auto_map_ignore_oid_labels = [
            'personal care',
            'dog bed',
            'caterpillar',
            'porcupine',
            'pineapple',
            'toilet paper',
            'bowling equipment',
            'cowboy hat',
            'scarf',
            'cattle',
            'bookcase',
            'cake stand',
            'carnivore',
            'beard',
            'seahorse',
            'bust',
            'wheelchair',
            'pizza cutter',
            'hot dog',
            'cat furniture',
            'measuring cup',
            'cupboard',
            'cart',
            'carrot',
            'human beard',
            'bottle opener',
            'training bench'
        ]

        for coco_label in coco_labels_filter:
            similar_labels = [oid_label for oid_label in oid_labels if coco_label in oid_label]
            for oid_label in similar_labels:
                if oid_label != coco_label and oid_label in auto_map_ignore_oid_labels:
                    continue
                oid_sub_labels_to_coco_id[oid_label] = coco_labels_to_id[coco_label]

                if oid_label != coco_label:
                    print ('Auto mapping labels: {} => {}'.format(oid_label, coco_label))

        if self.use_all_classes:
            oid_sub_labels_to_coco_id['houseplant']     = coco_labels_to_id['potted plant']
            oid_sub_labels_to_coco_id['mobile phone']   = coco_labels_to_id['cell phone']
            oid_sub_labels_to_coco_id['ball']           = coco_labels_to_id['sports ball']
            oid_sub_labels_to_coco_id['table']          = coco_labels_to_id['dining table']
            oid_sub_labels_to_coco_id['kitchen & dining room table'] = coco_labels_to_id['dining table']
            oid_sub_labels_to_coco_id['television']     = coco_labels_to_id['tv']
            oid_sub_labels_to_coco_id['hair dryer']     = coco_labels_to_id['hair drier']
            oid_sub_labels_to_coco_id['ski']            = coco_labels_to_id['skis']
            oid_sub_labels_to_coco_id['flying disc']    = coco_labels_to_id['frisbee']
            oid_sub_labels_to_coco_id['microwave oven'] = coco_labels_to_id['microwave']
            oid_sub_labels_to_coco_id['doughnut']       = coco_labels_to_id['donut']
            oid_sub_labels_to_coco_id['jet ski']        = coco_labels_to_id['skis']
            oid_sub_labels_to_coco_id['cattle']         = coco_labels_to_id['cow']
        else:
            pass
            # oid_sub_labels_to_coco_id['hair dryer']     = coco_labels_to_id['hair drier']

        filtered_annotations = {}
        for k in self.annotations:
            img_ann = self.annotations[k]

            filtered_boxes = []
            for ann in img_ann['boxes']:
                cls_id      = ann['cls_id']
                oid_label  = self.id_to_labels[cls_id]
                oid_label  = str.lower(oid_label)

                if oid_label in oid_sub_labels_to_coco_id:
                    ann['cls_id'] = oid_sub_labels_to_coco_id[oid_label]
                    filtered_boxes.append(ann)

            if len(filtered_boxes) > 0:
                filtered_annotations[k] = {'w': img_ann['w'], 'h': img_ann['h'], 'boxes': filtered_boxes}

        coco_id_to_labels = dict([(coco_labels_to_id[k], k) for k in coco_labels_to_id])
        return coco_id_to_labels, filtered_annotations


class ExtraOpenImagesGenerator(StatsOpenImageGenerator):
    def __init__(self,
                 coco_data_dir='/media/work/image/coco',
                 uniform_label_distribution=False,
                 **kwargs):
        super(ExtraOpenImagesGenerator, self).__init__(**kwargs)

        if uniform_label_distribution:
            self.annotations = self.__set_data_uniform_distribution()

        subset = kwargs['subset']
        if subset == 'validation':
            subset = 'val'

        print ('{}: {} annotations'.format(subset, len(self.annotations)))

        from ..preprocessing.coco import CocoGenerator

        # add negative examples
        negative_generator = CocoGenerator(
            coco_data_dir,
            '{}2017'.format(subset),
            batch_size=1
        )

        print('Adding {} negative examples'.format(len(self.annotations)))

        permutation = range(0, negative_generator.size())
        random.shuffle(permutation)
        permutation = permutation[:len(self.annotations)]

        self.negative_img_ids_to_img_paths = {}
        for image_index in permutation:
            image_info = negative_generator.coco.loadImgs(negative_generator.image_ids[image_index])[0]
            img_path = os.path.join(negative_generator.data_dir, 'images', negative_generator.set_name,
                                    image_info['file_name'])
            img_id = str(uuid.uuid4())

            self.annotations[img_id] = {'w': image_info['width'], 'h': image_info['height'], 'boxes': []}
            self.negative_img_ids_to_img_paths[img_id] = img_path

        self.id_to_image_id = dict([(i, k) for i, k in enumerate(self.annotations)])

    def image_path(self, image_index):
        img_id = self.id_to_image_id[image_index]

        if img_id in self.negative_img_ids_to_img_paths:
            return self.negative_img_ids_to_img_paths[img_id]

        return OpenImagesGenerator.image_path(self, image_index)

    def __set_data_uniform_distribution(self):
        if len(self.id_to_labels) == 1:
            return self.annotations

        stats_per_label = dict([(self.id_to_labels[i], 0) for i in range(len(self.id_to_labels))])

        for f in self.annotations.iteritems():
            for ann in f[1]['boxes']:
                stats_per_label[self.id_to_labels[ann['cls_id']]] += 1

        min_label = min(stats_per_label, key=stats_per_label.get)
        min_cnt = stats_per_label[min_label]

        stats_per_label_limit = dict([(self.id_to_labels[i], 0) for i in range(len(self.id_to_labels))])

        filtered_annotations = {}
        for f in self.annotations.iteritems():
            k = f[0]
            img_ann = f[1]

            filtered_boxes = []
            for ann in f[1]['boxes']:
                cnt = stats_per_label_limit[self.id_to_labels[ann['cls_id']]]

                if cnt < min_cnt:
                    stats_per_label_limit[self.id_to_labels[ann['cls_id']]] = cnt + 1
                    filtered_boxes.append(ann)

            if len(filtered_boxes) > 0:
                filtered_annotations[k] = {'w': img_ann['w'], 'h': img_ann['h'], 'boxes': filtered_boxes}

        return filtered_annotations

    def name_to_label(self, name):
        pass

    def stats(self):
        super(ExtraOpenImagesGenerator, self).stats()

        r = []
        for f in [f for f in self.annotations.iteritems() if len(f[1]['boxes']) > 1]:
            img_classes = set([ann['cls_id'] for ann in f[1]['boxes']])
            if len(img_classes) > 1:
                r.append(f)

        print ('there are {} images with at least 2 types of boxes'.format(len(r)))


if __name__ == '__main__':
    test_cocoid = True

    if not test_cocoid:
        from keras_retinanet.utils.transform import random_transform_generator

        transform_generator = random_transform_generator(flip_x_chance=0.5)
        generator = ExtraOpenImagesGenerator(
            main_dir='/media/work2/OpenImages',
            subset='train',
            version='2018_04',
            labels_filter=['Nail'],
            fixed_labels=True,
            uniform_label_distribution=False,
            annotation_cache_dir='/media/work2/OpenImages/2018_04',
            transform_generator=transform_generator
        )
    else:
        generator = CocoOpenImagesGenerator(
            main_dir='/media/work2/OpenImages',
            subset='train',
            version='v4',
            annotation_cache_dir='/media/work2/OpenImages/2018_04',
        )
    generator.load_image(608173)
    generator.stats()
