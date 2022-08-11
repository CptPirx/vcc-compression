# %%
# !pip install simplejpeg==1.6.4 objsize==0.3.3
# %%
SELECTED_GPUS = [0]  # which GPUs to use

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu_number) for gpu_number in SELECTED_GPUS])

import tensorflow as tf

tf.get_logger().setLevel('INFO')

# assert len(tf.config.list_physical_devices('GPU')) > 0
#
# GPUS = tf.config.experimental.list_physical_devices('GPU')
# for gpu in GPUS:
#     tf.config.experimental.set_memory_growth(gpu, True)

import cv2
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import sys
import logging
from datetime import datetime

from objsize import get_deep_size
from simplejpeg import decode_jpeg, encode_jpeg

SHT_B_DIR = '/data/shanghai_tech_cc/part_B_final'
SHT_B_TRAIN_COUNT = 400
SHT_B_TEST_COUNT = 316
SHT_B_VAL_RATIO = 0.1
VAL_RANDOM_SEED = 42
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)


# Set up logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# stdout logger
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

# File logger
# fh = logging.FileHandler(f'compressed_results/data_compression_{datetime.now()}.log')
# fh.setLevel(logging.INFO)
# logger.addHandler(fh)


# %%
def get_train_and_validation_indices():
    val_count = int(SHT_B_TRAIN_COUNT * SHT_B_VAL_RATIO)
    random.seed(VAL_RANDOM_SEED)
    all_indices = list(range(1, SHT_B_TRAIN_COUNT + 1))
    val_indices = random.sample(all_indices, val_count)
    train_indices = [item for item in all_indices if item not in val_indices]
    return np.array(train_indices, dtype=int), np.array(val_indices, dtype=int)


class VCCSequence(tf.keras.utils.Sequence):
    def __init__(self, split, config, save_cache=True):
        self.split = split
        self.batch_size = config['batch_size']
        self.cache_dir = config['dir']
        self.recreate_cache = config['recreate']
        self.target_compression = config['target_compression']
        self.save_cache = save_cache
        self.visualize = config['visualize']
        self.train_indices, self.val_indices = get_train_and_validation_indices()
        self.directory = 'test_data' if self.split == 'test' else 'train_data'
        if self.split == 'train':
            self.permutation = np.random.permutation(self.train_indices)
        elif self.split == 'val':
            self.permutation = self.val_indices
        else:  # test
            self.permutation = np.arange(1, SHT_B_TEST_COUNT + 1, dtype=int)

    def __len__(self):
        return math.ceil(len(self.permutation) / self.batch_size)

    def on_epoch_end(self):
        if self.split == 'train':
            self.permutation = np.random.permutation(self.train_indices)

    def _get_info_cache_path(self, info_index):
        return os.path.join(self.cache_dir, '%s_%.2f_%d.pkl' % (
            self.split,
            self.target_compression,
            info_index
        ))

    def __getitem__(self, index):
        density_maps = []
        compressed_images = []
        sizes = []
        tmp_image_path = 'tmp.jpg'
        indices = self.permutation[index * self.batch_size:(index + 1) * self.batch_size]
        for info_index in indices:
            info_cache_path = self._get_info_cache_path(info_index)
            if not self.recreate_cache and os.path.exists(info_cache_path):
                with open(info_cache_path, 'rb') as cache_file:
                    contents = pickle.load(cache_file)
                density_map = contents['density_map']
                image = contents['image']
                size = contents['size']
                density_maps.append(density_map)
                compressed_images.append(image)
                sizes.append(size)
            else:
                image_path = os.path.join(SHT_B_DIR, self.directory, 'images', 'IMG_%d.jpg' % info_index)
                image = plt.imread(image_path, format='jpeg')

                encoded_image = encode_jpeg(image, int(self.target_compression * 100))
                compressed_image_size = get_deep_size(encoded_image)
                sizes.append(compressed_image_size)
                compressed_image = decode_jpeg(encoded_image)
                compressed_images.append(compressed_image)

                density_file_path = os.path.join(
                    SHT_B_DIR, self.directory, 'ground_truth', 'IMG_%d_sigma4.h5' % info_index
                )
                with h5py.File(density_file_path, 'r') as h5_file:
                    density_map = h5_file['density'][()]
                density_maps.append(density_map)

                if self.save_cache:
                    with open(info_cache_path, 'wb') as cache_file:
                        pickle.dump({
                            'image': compressed_image,
                            'density_map': density_map,
                            'size': compressed_image_size,
                        }, cache_file)

                if self.visualize:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 20))
                    ax1.imshow(image)
                    ax2.imshow(compressed_image)
                    ax1.set_title('original')
                    ax2.set_title('compressed')
                    ax1.axis('off')
                    ax2.axis('off')
                    plt.show()

        output_images = np.array(compressed_images, dtype=np.float32)
        output_density_maps = np.expand_dims(
            np.array(density_maps, dtype=np.float32),
            axis=-1
        )
        return output_images, output_density_maps, np.mean(np.array(sizes))


# %%
def create_normal_compressed_dataset(config):
    if not os.path.exists(config['dir']):
        os.makedirs(config['dir'])
    for split in ['train', 'val', 'test']:
        sizes = []
        for i, (_, _, size) in enumerate(VCCSequence(split, config)):
            sys.stdout.write('\r%s %d' % (split, i + 1))
            sys.stdout.flush()
            sizes.append(size)
        logger.info('Avg. size: %.2f' % np.mean(np.array(sizes)))


# %%
config = {
    'dir': 'normal_compressed_shtb',
    'batch_size': 1,
    'recreate': True,
    'target_compression': 1.0,
    'visualize': False,
}

target_compression = range(75, 0, -5)

for i, c in enumerate(target_compression):
    logger.info('_' * 20)
    logger.info(f'Run {i + 1}/{len(target_compression)}')
    logger.info(f'Target compression: {c / 100}%')
    run_config = config.copy()
    run_config['target_compression'] = config['target_compression'] * (c / 100)
    run_config['dir'] += f'_{c}'
    logger.info(f'Config: {run_config}')
    create_normal_compressed_dataset(run_config)
