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
import base64

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
import time
import matplotlib.gridspec as gridspec

from io import BytesIO
from PIL import Image, ImageDraw, ImageChops
from torchvision import datasets, models, transforms
from colorama import Fore, Back, Style, init as color_init

color_init()

rospy.init_node('chris_magic', anonymous=True)
pub = rospy.Publisher('/dRt', String, queue_size=1000)

pub1 = rospy.Publisher('/im1', String, queue_size=1000)
pub2 = rospy.Publisher('/im2', String, queue_size=1000)
pub3 = rospy.Publisher('/im3', String, queue_size=1000)
pub4 = rospy.Publisher('/im_info', String, queue_size=1000)

current_image = Image.open('empty.jpeg')
s1_image = Image.open('empty.jpeg')
sub_image = Image.open('empty.jpeg')

calc_reward_real_time = False  # use the current img as s2 to calc dRt real_time?

update_s1_as_current_auto = False  #  automatically set current as s1_image, after calc the deltaRt?

rt_str = "haha"

input_size = 240
sampling_size = 200
conf_level = 0.999

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
task_dim = 6
R = R_model_nn.R_model_NN(task_dim, device, _sampling_size=sampling_size, _conf_level=conf_level, _lr=0.001)
R.load_weights(torch.load('../results/R_params0810_table12345mix'))
print('R params loaded!')
R = R.to(device)
print(R)

torch.backends.cudnn.benchmark = True

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
    ]
)

fig = plt.figure()
fig.canvas.set_window_title('InMaxEnt_IRL demo v0.1')
gs = gridspec.GridSpec(2, 2)

image_str=""

inner_count = 0

input("Press Enter to continue...")

def pilImg2str(im):
    output_buffer = BytesIO()
    # conver to base64 encodes
    im.save(output_buffer, format="JPEG")
    img_str = base64.b64encode(output_buffer.getvalue())
    return img_str.decode("utf-8")

def conver2str(np_vector):
    rs = ""
    for i in range(np_vector.shape[0]):
        rs += (str(0 - np_vector[i]))[0:5]+" "
    return rs[0:(len(rs)-1)]

def callback(data):
    '''
    subscribe topics publishing image base64 string, without header.
    :param data:
    :return:
    '''
    global image_str
    rospy.loginfo(rospy.get_caller_id() + "img_str %s", data.data[0:10])
    if data.data is "" or data.data is "h":
        print('run data.data')
        displayGUI(s1_image, current_image, rt_str)
        return

    global current_image
    # process the image data, and publish an error msg.
    image_str = data.data
    binary_data = base64.b64decode(image_str)
    image_data = BytesIO(binary_data)
    current_image = Image.open(image_data)

    # current_image = Image.open(BytesIO(base64.b64decode(image_str)))
    #  display current image
    if calc_reward_real_time:
        get_d_Rt(1)
    else: # display GUI comes to publish img string
        displayGUI(s1_image, current_image, rt_str, sub_image)


def callback2(data):
    '''
    subscribe to /dRt_CMD, when got 0: mark s1, when got 1: using current_image as s2, s2-s1, and get the deltaRewards
    :param data: string 0/1
    :return:
    '''
    samples = torch.zeros(1, 1, input_size, input_size)
    global s1_image
    global rt_str

    if data.data is "0": #  mark s1_image, it's 0 only at the initial stage, when put in practice
        s1_image = current_image
        displayGUI(s1_image, current_image, "")
    elif data.data is "1": # mark current image as s2_image, calculate dS, and output dRt
        print('get_drt')
        get_d_Rt(1)

def callback3(data):
    '''
    subscribe to /dRt_CMD, when got 0: mark s1, when got 1: using current_image as s2, s2-s1, and get the deltaRewards
    :param data: string 0/1
    :return:
    '''
    samples = torch.zeros(1, 1, input_size, input_size)
    global s1_image
    global rt_str

    if data.data is "0": #  mark s1_image, it's 0 only at the initial stage, when put in practice
        s1_image = current_image
        displayGUI(s1_image, current_image, "")
    elif data.data is "1": # mark current image as s2_image, calculate dS, and output dRt
        print('get_drt')
        get_d_Rt(1)

def get_d_Rt(max_count):
    '''
    calc one delta Rt, and update s1_image, rt_str, and also the display GUI
    :return:
    '''
    global s1_image
    global rt_str
    global sub_image
    global inner_count
    inner_count += 1
    if inner_count >= max_count:
        inner_count = 0
        samples = torch.zeros(1, 1, input_size, input_size)
        sub_image = ImageChops.subtract_modulo(current_image, s1_image)
        # now transform to py tensor and feed to the network
        samples[0] = transform(sub_image)
        delta_Rewards = R.forward(samples.to(device), False)
        # delta_rewards[0] to string
        rt_str =conver2str(delta_Rewards[0].detach().to('cpu').numpy())
        # now publish this delta_reward
        pub.publish(rt_str)
        # now re mark current image as s1 image
        print('I will publish my results.')
        if update_s1_as_current_auto is True:
            s1_image = current_image

    displayGUI(s1_image, current_image, rt_str, sub_image)


def displayGUI(im1, im2, delta_Rt_string, sub_img=Image.open('empty.jpeg')):
    '''
    using matplotlib to plot results., when receiving current_img, cmd 0, or cmd 1, it will update.
    :param im1: the s1 image, pillow image.
    :param im2: the s2 image, pillow image.
    :param delta_Rt_string: the result delta_Rt_string
    :return:
    '''
    # publish them all
    # print("published!")
    pub1.publish(pilImg2str(im1))
    # pub2.publish(pilImg2str(im2))
    pub3.publish(pilImg2str(sub_img))
    pub4.publish(delta_Rt_string)

    # plt.clf()
    # fig.suptitle("InMaxEnt_IRL demo v0.1")
    # plt.subplot(gs[0])
    # plt.imshow(im1, cmap='gray')
    #
    # plt.subplot(gs[1])
    # plt.imshow(im2, cmap='gray')
    #
    # plt.subplot(gs[2])
    # plt.imshow(sub_img, cmap='gray')
    #
    # ax4 = plt.subplot(gs[3])
    # ax4.axis('off')
    # ax4.text(-0.1, 0.2, delta_Rt_string, color="blue", wrap=True, fontsize=12)
    #
    # plt.pause(1e-17)
    #
    # plt.show(block=False)


def listener():
    rospy.Subscriber("/img_str", String, callback)
    rospy.Subscriber("/dRt_CMD", String, callback2)
    rospy.Subscriber("/froglake/dRt_CMD", String, callback3)
    rospy.spin()

if __name__ == '__main__':
    # load listener
    print("start working...")
    # testing the display GUI
    # for i in range(100):
    #     displayGUI(s1_image, current_image, "sssss deltaRts    " + str(i))
    listener()
