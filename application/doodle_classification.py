import fastai
import torch
from fastai.vision.all import *
# Input tensors get tagged as `TensorImageBW`, and they keep that tag even after going through the model.
# I'm not sure how you're supposed to drop that tag, but this works around a type dispatch error.
TensorImageBW.register_func(F.cross_entropy, TensorImageBW, TensorCategory)

import sys
import numpy as np
from matplotlib import pyplot as plt
import random

