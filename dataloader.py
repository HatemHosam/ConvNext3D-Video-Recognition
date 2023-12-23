import os
import pytorchvideo
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)

class KineticsDataModule(pytorch_lightning.LightningDataModule):

  # Dataset configuration
  # Insert the data path in here
  _DATA_PATH = "/data/i5O/k600/"
  _CLIP_DURATION = 16  # Duration of sampled clip for each video
  _BATCH_SIZE = 16
  _NUM_WORKERS = 0  # Number of parallel processes fetching data

  def train_dataloader(self):
    """
    Create the Kinetics train partition from the list of video labels
    in {self._DATA_PATH}/train
    """
    train_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    Resize((360,204)),
                    UniformTemporalSubsample(16),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(244),
                    RandomHorizontalFlip(p=0.5),
                  ]
                ),
              ),
            ]
        )
    train_dataset = pytorchvideo.data.Kinetics(
        data_path=os.path.join(self._DATA_PATH, "train"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION),
        transform=train_transform
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=self._BATCH_SIZE,
        num_workers=self._NUM_WORKERS,
    )

  def val_dataloader(self):
    """
    Create the Kinetics validation partition from the list of video labels
    in {self._DATA_PATH}/val
    """
    val_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    Resize((360,204)),
                    UniformTemporalSubsample(16),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                  ]
                ),
              ),
            ]
        )
    val_dataset = pytorchvideo.data.Kinetics(
        data_path=os.path.join(self._DATA_PATH, "val"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self._CLIP_DURATION),
        transform=val_transform
    )
    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=self._BATCH_SIZE,
        num_workers=self._NUM_WORKERS,
    )
