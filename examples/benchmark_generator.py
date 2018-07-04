from tqdm import tqdm
from keras_retinanet.utils.transform import random_transform_generator

transform_generator = random_transform_generator(flip_x_chance=0.5)

if False:
    from keras_retinanet.preprocessing.labelbox_generator import LabelBoxGenerator

    train_generator = LabelBoxGenerator(
        main_dir='/media/work/image/enedis_construction',
        json_filename='labelbox_v2.json',
        subset='train',
        add_negative_examples=False,
        transform_generator=transform_generator,
        batch_size=1
    )
else:
    from keras_retinanet.preprocessing.coco import CocoGenerator

    train_generator = CocoGenerator(data_dir='/media/work/image/coco',
                                    set_name='train2017',
                                    transform_generator=transform_generator,
                                    batch_size=1
                                    )


def benchmark_gen():
    while True:
        a = train_generator.next()
        if a is None:
            break
        yield a


for _ in tqdm(benchmark_gen()):
    pass
