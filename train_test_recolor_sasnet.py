# %%
SELECTED_GPUS = [0, 1, 2, 3]  # which GPUs to use

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu_number) for gpu_number in SELECTED_GPUS])

import h5py
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.io
import skimage.transform
import sys
import time
import torch
import torchvision
import yaml
from sklearn.metrics import mean_squared_error
import logging
from datetime import datetime

from PIL import Image
from sasnet_model import SASNet
from torchinfo import summary

from objsize import get_deep_size
from simplejpeg import decode_jpeg, encode_jpeg

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SHT_B_TRAIN_COUNT = 400
SHT_B_TEST_COUNT = 316
SHT_B_DIR = 'data/shanghai_tech_cc/part_B_final'
SHT_B_VAL_RATIO = 0.1
VAL_RANDOM_SEED = 42
SHT_B_SASNET_PATH = 'models/sasnet/pretrained/SHHB.pth'

# Set up logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# stdout logger
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

# File logger
fh = logging.FileHandler(f'color_results/{datetime.now()}.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)


# %%
class Args:
    def __init__(self, block_size):
        self.block_size = block_size


def get_model():
    args = Args(32)
    model = SASNet(args=args).cuda()
    model.load_state_dict(torch.load(SHT_B_SASNET_PATH))
    return model


# %%
def get_train_and_validation_indices():
    val_count = int(SHT_B_TRAIN_COUNT * SHT_B_VAL_RATIO)
    random.seed(VAL_RANDOM_SEED)
    all_indices = list(range(1, SHT_B_TRAIN_COUNT + 1))
    val_indices = random.sample(all_indices, val_count)
    train_indices = [item for item in all_indices if item not in val_indices]
    return np.array(train_indices, dtype=int), np.array(val_indices, dtype=int)


class CrowdCountingDataset(torch.utils.data.Dataset):
    def __init__(self, split, model_input_size):
        self.split = split
        self.model_input_size = model_input_size
        self.directory = 'test_data' if self.split == 'test' else 'train_data'
        train_indices, val_indices = get_train_and_validation_indices()
        if self.split == 'test':
            self.indices = np.arange(1, SHT_B_TEST_COUNT + 1, dtype=int)
        elif self.split == 'train':
            self.indices = np.array(train_indices)
        else:  # val
            self.indices = np.array(val_indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        true_index = self.indices[index]

        image_path = os.path.join(SHT_B_DIR, self.directory, 'images', 'IMG_%d.jpg' % true_index)
        image = Image.open(image_path).convert('RGB')
        # image = image.resize((self.model_input_size[1], self.model_input_size[0]))
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Normalize(
                    mean=[0.445],
                    std=[0.269]
            ),
        ])
        normalized_image = transform(image)
        normalized_image = normalized_image.expand(3, 768, 1024)
        normalized_image = normalized_image.float()

        density_file_path = os.path.join(
                SHT_B_DIR,
                self.directory,
                'ground_truth',
                'IMG_%d_sigma4.h5' % true_index
        )
        with h5py.File(density_file_path, 'r') as h5_file:
            density_map = h5_file['density'][()]

        resized_density_map = skimage.transform.resize(density_map, self.model_input_size)
        resized_density_map *= np.sum(density_map) / np.sum(resized_density_map)
        resized_density_map = torch.tensor(resized_density_map)
        resized_density_map = resized_density_map.float()
        resized_density_map = torch.unsqueeze(resized_density_map, 0)

        return normalized_image, resized_density_map


# %%
def calculate_counts(density_maps):
    if torch.is_tensor(density_maps):
        return torch.sum(density_maps, (1, 2, 3))
    else:  # np.ndarray
        return np.sum(density_maps, axis=(1, 2, 3))


def calculate_mae_mse(model, data_loader):
    # TODO multi-GPU
    model.eval()
    running_absolute_errors = None
    running_mean_squared_errors = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            images, density_maps = data
            images = images.cuda()
            predictions = model(images)
            predictions = predictions.cpu().detach().numpy()
            # https://github.com/TencentYoutuResearch/CrowdCounting-SASNet/blob/main/main.py#L87
            predictions /= 1000
            absolute_errors = torch.abs(calculate_counts(density_maps) - calculate_counts(predictions))

            ms_errors = (torch.Tensor(calculate_counts(predictions)) - calculate_counts(density_maps)) * (
                    torch.Tensor(calculate_counts(predictions)) - calculate_counts(density_maps))

            if running_absolute_errors is None:
                running_absolute_errors = absolute_errors
                running_mean_squared_errors = ms_errors
            else:
                running_absolute_errors = torch.cat((running_absolute_errors, absolute_errors))
                running_mean_squared_errors = torch.cat((running_mean_squared_errors, ms_errors))
    mae = torch.mean(running_absolute_errors).item()
    mse = float(np.sqrt(torch.mean(running_mean_squared_errors).item()))
    model.train()
    return mae, mse


# %%
def get_model_path(config):
    return 'color_results/sasnet_%d_%d_%s.pth' % (config['input_size'][0], config['input_size'][1], config['version'])


def train(config):
    # create datasets and data loaders
    train_dataset = CrowdCountingDataset('train', config['input_size'])
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['batch_size']
    )
    val_dataset = CrowdCountingDataset('val', config['input_size'])
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['batch_size']
    )
    test_dataset = CrowdCountingDataset('test', config['input_size'])
    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['batch_size']
    )

    model = get_model()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    criterion = torch.nn.MSELoss()

    # training loop
    best_val_mae = float('inf')
    for epoch in range(config['epochs']):
        start_time = time.perf_counter()
        running_loss = 0.0
        running_absolute_errors = None
        running_mean_squared_errors = None
        for i, data in enumerate(train_loader):
            # load data
            images, density_maps = data
            images = images.cuda()
            density_maps = density_maps.cuda()

            # train the model
            optimizer.zero_grad()
            predicted_density_maps = model(images)
            predicted_density_maps /= 1000
            loss = criterion(predicted_density_maps, density_maps)
            loss.backward()
            optimizer.step()

            # update training metrics
            absolute_errors = torch.abs(calculate_counts(density_maps) - calculate_counts(predicted_density_maps))

            mean_squared_errors = (calculate_counts(predicted_density_maps) - calculate_counts(density_maps)) * (
                    calculate_counts(predicted_density_maps) - calculate_counts(density_maps)).sum().data

            if running_absolute_errors is None:
                running_absolute_errors = absolute_errors
                running_mean_squared_errors = mean_squared_errors
            else:
                running_absolute_errors = torch.cat((running_absolute_errors, absolute_errors))
                running_mean_squared_errors = torch.cat((running_mean_squared_errors, mean_squared_errors))
            running_loss += loss.item()

        # update training metrics
        train_mae = torch.mean(running_absolute_errors).item()
        train_mse = float(np.sqrt(torch.mean(running_mean_squared_errors).item()))

        # update val metrics
        val_mae, val_mse = calculate_mae_mse(model, val_loader)
        saved = False

        # update test metrics
        test_mae, test_mse = calculate_mae_mse(model, test_loader)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), get_model_path(config))
            saved = True

            result = {'loss': running_loss,
                      'train_mae': train_mae,
                      'train_mse': train_mse,
                      'val_mae': val_mae,
                      'val_mse': val_mse,
                      'test_mae': test_mae,
                      'test_mse': test_mse}
            with open(get_model_path(config)[:-4] + '_results.yaml', 'w') as f:
                yaml.dump(result, f)

        end_time = time.perf_counter()

        logger.info('Epoch: %d/%d; Loss: %.2e; Train MAE: %.2f; Val MAE: %.2f; Test MAE: %.2f; took %.2fms%s' % (
            epoch + 1,
            config['epochs'],
            running_loss,
            train_mae,
            val_mae,
            test_mae,
            (end_time - start_time) * 100,
            '; saved' if saved else ''
        ))


# %%
config = {
    'batch_size': 1 * len(SELECTED_GPUS),
    'input_size': (768, 1024),
    'lr': 1e-5,
    'wd': 1e-4,
    'epochs': 100,
    'version': 'v1',
}

logger.info('_' * 20)
run_config = config.copy()
logger.info(f'Config: {run_config}')
train(run_config)

# One time experiment to save image sizes
# image = Image.open('../..//data/shanghai_tech_cc/part_B_final/train_data/images/IMG_1.jpg').convert('L')
# image = np.asarray(image)
# compressed_image_size = image.nbytes / 1000
#
# # encoded_image = encode_jpeg(image, 75)
# # encoded_image = encode_jpeg(np.asarray(image), 75)
# print(compressed_image_size)
