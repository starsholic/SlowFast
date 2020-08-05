import logging
import numpy as np
import torch

from . import moviegraph_helper as moviegraph_helper
from . import cv2_transform as cv2_transform
from . import transform as transform
from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register()
class MovieGraph(torch.utils.data.Dataset):
    def __init__(self):
        pass
    def __len__(self):
        pass
    def __getitem__(self):
        pass