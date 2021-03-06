import sys
sys.path.insert(0,'../')
import numpy as np
import rospy
import roslib
from std_msgs.msg import String
import t_utils
import G_model
import G2_model
import R_model_nn
import torch
import gc
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageChops
import base64
from scipy.misc import imsave, imread, imresize
from imageio import imread as imread2
import io
from io import BytesIO

from colorama import Fore, Back, Style, init as color_init

color_init()

base64_str = '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'

byte_data = base64.b64decode(base64_str)
image_data = BytesIO(byte_data)
img = Image.open(image_data)

