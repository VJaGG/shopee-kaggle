import os
import time
import scipy.io
from datetime import datetime
from timeit import default_timer as timer

import copy
import pickle
import random

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import cv2
import albumentations as A
import matplotlib.pyplot as plt

import yaml
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("/home/data_normal/abiz/wuzhiqiang/wzq/shopee/code_v3/pytorch-image-models")
import timm
from transformers import AutoConfig, AutoModel

def seed_py(seed):
    random.seed(seed)
    np.random.seed(seed)
    return seed