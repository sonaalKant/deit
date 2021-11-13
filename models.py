# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from vision_transformer import *
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

