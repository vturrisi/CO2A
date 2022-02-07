import random
import re
from os import listdir
from os.path import join

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# converts string to integers when possible
def atoi(text):
    return int(text) if text.isdigit() else text


# applies atoi to a string
def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", str(text))]


# applies the Gaussian blur augmentation from SimCLR https://arxiv.org/abs/2002.05709
class GaussianBlur(object):
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


# returns dataset mean and std
def get_mean_std():
    mean = [0.4345, 0.4051, 0.3775]
    std = [0.2768, 0.2713, 0.2737]
    return mean, std


class VideoDataset(Dataset):
    def __init__(
        self,
        folder,
        n_frames=16,
        n_clips=1,
        frame_size=224,
        augmentations=None,
        augmentations_params=None,
        reorder_shape=True,
        normalize=True,
    ):
        super().__init__()

        self.base_dir = folder

        if augmentations is None:
            augmentations = []
        if augmentations_params is None:
            augmentations_params = {}

        self.augmentations = augmentations
        self.augmentations_params = augmentations_params

        self.n_frames = n_frames
        self.n_clips = n_clips
        self.reorder_shape = reorder_shape
        self.normalize = normalize
        self.mean, self.std = get_mean_std()

        if isinstance(frame_size, int):
            self.frame_size = (frame_size, frame_size)
        else:
            self.frame_size = frame_size

        self.videos_with_class = []
        self.classes = sorted(listdir(folder))

        # select all videos with enough frames
        for y, c in enumerate(self.classes):
            d = join(self.base_dir, c)
            videos = listdir(d)
            for video in videos:
                video = join(d, video)
                if len(self.find_frames(video)) >= n_frames:
                    self.videos_with_class.append((video, y))

    # implements __len(self)__ from Dataset class
    def __len__(self):
        return len(self.videos_with_class)

    # checks if input is image
    def is_img(self, f):
        return str(f).lower().endswith("jpg") or str(f).lower().endswith("jpeg")

    # selects frames from input sequence
    def find_frames(self, video):
        frames = [join(video, f) for f in listdir(video) if self.is_img(f)]
        return frames

    # select keypoints fromi input sequence
    def find_keypoints(self, video):
        keypoints = [join(video, kp) for kp in listdir(video) if kp.lower().endswith("json")]
        return keypoints

    # handles the case where tensor was converted to gray scale
    def maybe_fix_gray(self, tensor):
        if tensor.size(0) == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor

    # generates the pipeline of transformations that we apply to an image
    def gen_transformation_pipeline(self):
        if self.frame_size[0] == 224:
            s = (256, 256)
        elif self.frame_size[0] == 112:
            s = (128, 128)
        else:
            raise Exception("Size is not supported")

        transformations = [(TF.resize, s)]
        if "spatial" in self.augmentations:
            dummy = Image.new("RGB", s)
            i, j, h, w = transforms.RandomCrop.get_params(dummy, self.frame_size)
            transformations.append((TF.resized_crop, i, j, h, w, self.frame_size))
        else:
            transformations.append((TF.center_crop, self.frame_size))

        if "color" in self.augmentations and random.random() < 0.8:
            color_jitter = transforms.ColorJitter(0.15, 0.15, 0.15, 0.05)
            (
                fn_idx,
                brightness_factor,
                contrast_factor,
                saturation_factor,
                hue_factor,
            ) = color_jitter.get_params(
                color_jitter.brightness,
                color_jitter.contrast,
                color_jitter.saturation,
                color_jitter.hue,
            )
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    aug, param = TF.adjust_brightness, brightness_factor
                elif fn_id == 1 and contrast_factor is not None:
                    aug, param = TF.adjust_contrast, contrast_factor
                elif fn_id == 2 and saturation_factor is not None:
                    aug, param = TF.adjust_saturation, saturation_factor
                elif fn_id == 3 and hue_factor is not None:
                    aug, param = TF.adjust_hue, hue_factor
                transformations.append((aug, param))

        if "horizontal" in self.augmentations and random.random() < 0.5:
            transformations.append((TF.hflip,))

        if "gaussian" in self.augmentations:
            transformations.append((GaussianBlur(),))

        if "gray" in self.augmentations and random.random() < 0.2:
            transformations.append((TF.to_grayscale,))

        transformations.append((TF.to_tensor,))
        transformations.append((self.maybe_fix_gray,))

        if self.normalize and self.mean is not None:
            transformations.append((TF.normalize, self.mean, self.std))
        return transformations

    # applies the generated transformations to an image
    def apply_transforms(self, frame, transformations):
        for transform, *args in transformations:
            if isinstance(transform, tuple):
                print(transform)
            frame = transform(frame, *args)
        return frame

    # stacks the frame-level features into a video-level feature
    def convert_to_video(self, frames):
        transformations = self.gen_transformation_pipeline()

        tensors = []
        for frame in frames:
            frame = self.apply_transforms(frame, transformations)
            tensors.append(frame)

        tensors = torch.stack(tensors)
        tensors = tensors.reshape(self.n_clips, self.n_frames, *tensors.size()[1:])

        if self.reorder_shape:
            tensors = tensors.permute(0, 2, 1, 3, 4)
        return tensors

    # loads image from file
    def load_frame(self, path):
        frame = Image.open(path)
        return frame

    # generates random indices from sequence
    def get_random_indices(self, num_frames):
        indexes = np.sort(np.random.choice(num_frames, self.n_frames * self.n_clips, replace=True))
        return indexes

    # generates uniformly distributed indices from sequence
    def get_indices(self, num_frames):
        tick = num_frames / self.n_frames
        indexes = np.array(
            [int(tick / 2.0 + tick * x) for x in range(self.n_frames)]
        )  # pick the central frame in each segment
        return indexes

    # retrieves clip indices
    def get_indices_clips(self, num_frames):
        num_frames_clip = num_frames // self.n_clips
        indexes = self.get_indices(num_frames_clip)
        indexes = np.tile(indexes, self.n_clips)
        for i in range(self.n_clips):
            indexes[i * self.n_frames : (i + 1) * self.n_frames] += num_frames_clip * i
        return indexes

    # implements __getitem__ from Dataset class
    def __getitem__(self, index):
        video, y = self.videos_with_class[index]

        # find frames
        frame_paths = self.find_frames(video)
        frame_paths.sort(key=natural_keys)

        n_frames = len(frame_paths)
        if "temporal" in self.augmentations:
            indexes = self.get_random_indices(n_frames)
        else:
            indexes = self.get_indices_clips(n_frames)

        frames = []
        for i in indexes:
            frames.append(self.load_frame(frame_paths[i]))

        tensor = self.convert_to_video(frames)
        return tensor, y


# contrastive version of the video dataset: loads 2 target augmentations
class VideoDatasetContrastive:
    def __init__(
        self,
        folder,
        n_frames=16,
        n_clips=1,
        frame_size=224,
        augmentations=None,
        augmentations_params=None,
        reorder_shape=True,
        normalize=True,
    ):
        self.dataset = VideoDataset(
            folder=folder,
            n_frames=n_frames,
            n_clips=n_clips,
            frame_size=frame_size,
            augmentations=augmentations,
            augmentations_params=augmentations_params,
            reorder_shape=reorder_shape,
            normalize=normalize,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = []
        for i in range(2):
            tensor, y = self.dataset[index]
            data.append(tensor)
        return (*data, y)


# variant of video dataset: defines two distinct datasets for source and target
class VideoDatasetSourceAndTarget:
    def __init__(self, source_dataset, target_dataset):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def __len__(self):
        return max([len(self.source_dataset), len(self.target_dataset)])

    def __getitem__(self, index):
        source_index = index % len(self.source_dataset)
        source_data = self.source_dataset[source_index]

        target_index = index % len(self.target_dataset)
        target_data = self.target_dataset[target_index]
        return (source_index, *source_data, target_index, *target_data)


# prepares datasets
def prepare_datasets(
    source_dataset,
    target_dataset,
    val_dataset,
    source_augmentations=None,
    target_augmentations=None,
    source_augmentations_params=None,
    target_augmentations_params=None,
    n_frames=4,
    n_clips=4,
    frame_size=224,
    normalize=True,
    target_2_augs=False,
):
    source_dataset = VideoDataset(
        source_dataset,
        frame_size=frame_size,
        n_frames=n_frames,
        n_clips=n_clips,
        augmentations=source_augmentations,
        augmentations_params=source_augmentations_params,
        normalize=normalize,
    )
    if target_2_augs:
        dataset_class = VideoDatasetContrastive
    else:
        dataset_class = VideoDataset
    target_dataset = dataset_class(
        target_dataset,
        frame_size=frame_size,
        n_frames=n_frames,
        n_clips=n_clips,
        augmentations=target_augmentations,
        augmentations_params=target_augmentations_params,
        normalize=normalize,
    )
    source_n_target_dataset = VideoDatasetSourceAndTarget(source_dataset, target_dataset)

    val_dataset = VideoDataset(
        val_dataset,
        frame_size=frame_size,
        n_frames=n_frames,
        n_clips=n_clips,
        normalize=normalize,
    )

    return source_n_target_dataset, val_dataset


# prepare datasets in the source-only case
def prepare_datasets_source_only(
    source_dataset,
    val_dataset,
    augmentations=None,
    augmentations_params=None,
    n_frames=4,
    n_clips=4,
    frame_size=224,
    normalize=True,
):
    source_dataset = VideoDataset(
        source_dataset,
        frame_size=frame_size,
        n_frames=n_frames,
        n_clips=n_clips,
        augmentations=augmentations,
        augmentations_params=augmentations_params,
        normalize=normalize,
    )

    val_dataset = VideoDataset(
        val_dataset,
        frame_size=frame_size,
        n_frames=n_frames,
        n_clips=n_clips,
        normalize=normalize,
    )

    return source_dataset, val_dataset


# prepares dataloaders given input datasets
def prepare_dataloaders(train_dataset, val_dataset, batch_size=64, num_workers=4):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


# prepares datasets and dataloaders
def prepare_data(
    source_dataset,
    target_dataset,
    val_dataset,
    n_frames=16,
    n_clips=1,
    frame_size=224,
    normalize=True,
    source_augmentations=None,
    target_augmentations=None,
    source_augmentations_params=None,
    target_augmentations_params=None,
    target_2_augs=False,
    batch_size=64,
    num_workers=4,
):
    source_n_target_dataset, val_dataset = prepare_datasets(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        val_dataset=val_dataset,
        source_augmentations=source_augmentations,
        target_augmentations=target_augmentations,
        source_augmentations_params=source_augmentations_params,
        target_augmentations_params=target_augmentations_params,
        n_frames=n_frames,
        n_clips=n_clips,
        frame_size=frame_size,
        target_2_augs=target_2_augs,
        normalize=normalize,
    )
    source_n_target_loader, val_loader = prepare_dataloaders(
        source_n_target_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return source_n_target_loader, val_loader


# prepares datasets and dataloaders in the source-only case
def prepare_data_source_only(
    source_dataset,
    val_dataset,
    n_frames=16,
    n_clips=1,
    frame_size=224,
    normalize=True,
    augmentations=None,
    augmentations_params=None,
    batch_size=64,
    num_workers=4,
):
    source_dataset, val_dataset = prepare_datasets_source_only(
        source_dataset=source_dataset,
        val_dataset=val_dataset,
        augmentations=augmentations,
        augmentations_params=augmentations_params,
        n_frames=n_frames,
        n_clips=n_clips,
        frame_size=frame_size,
        normalize=normalize,
    )

    source_loader, val_loader = prepare_dataloaders(
        source_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return source_loader, val_loader
