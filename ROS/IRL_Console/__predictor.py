import sys
sys.path.insert(0,'../')
import numpy as np
import rospy
import roslib
from std_msgs.msg import String
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageChops

rospy.init_node('chris_magic', anonymous=True)
pub = rospy.Publisher('/dRt', String, queue_size=10)

current_image = Image.open('empty.png').convert('LA')
s1_image = Image.open('empty.png').convert('LA')
sub_image = Image.open('empty.png').convert('LA')

calc_reward_real_time = False  # use the current img as s2 to calc dRt real_time?

update_s1_as_current_auto = False  #  automatically set current as s1_image, after calc the deltaRt?

rt_str = ""

input_size = 240
sampling_size = 200
conf_level = 0.999

print('R params loaded!')

image_str=""

input("Press Enter to continue...")

def conver2str(np_vector):
    rs = ""
    for i in range(np_vector.shape[0]):
        rs += str(np_vector[i])+","
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
        displayGUI(s1_image, current_image, rt_str)
        return

    # process the image data, and publish an error msg.
    image_str = data.data
    binary_data = base64.b64decode(image_str)
    image_data = BytesIO(binary_data)
    img = Image.open(image_data)
    global current_image
    current_image = Image.open(BytesIO(base64.b64decode(image_str)))
    #  display current image
    if calc_reward_real_time:
        get_d_Rt()
    else:
        displayGUI(s1_image, current_image, rt_str, sub_image)


def callback2(data):
    '''
    subscribe to /dRt_CMD, when got 0: mark s1, when got 1: using current_image as s2, s2-s1, and get the deltaRewards
    :param data: string 0/1
    :return:
    '''
    # samples = torch.zeros(1, 1, input_size, input_size)
    global s1_image
    global rt_str

    if data.data is "0": #  mark s1_image, it's 0 only at the initial stage, when put in practice
        s1_image = current_image
        displayGUI(s1_image, current_image, "")
    elif data.data is "1": # mark current image as s2_image, calculate dS, and output dRt
        get_d_Rt()


def get_d_Rt():
    '''
    calc one delta Rt, and update s1_image, rt_str, and also the display GUI
    :return:
    '''
    global s1_image
    global rt_str
    global sub_image
    # samples = torch.zeros(1, 1, input_size, input_size)
    sub_img = ImageChops.subtract_modulo(current_image, s1_image)
    # now transform to py tensor and feed to the network
    # samples[0] = transform(sub_img)
    # delta_Rewards = R.forward(samples.to(device), False)
    # delta_rewards[0] to string
    rt_str = "some delta rewards"
    # now publish this delta_reward
    pub.publish(rt_str)
    # now re mark current image as s1 image
    if update_s1_as_current_auto is True:
        s1_image = current_image

    displayGUI(s1_image, current_image, rt_str, sub_img)


def displayGUI(im1, im2, delta_Rt_string, sub_img=Image.open('empty.png').convert('LA')):
    '''
    using matplotlib to plot results., when receiving current_img, cmd 0, or cmd 1, it will update.
    :param im1: the s1 image, pillow image.
    :param im2: the s2 image, pillow image.
    :param delta_Rt_string: the result delta_Rt_string
    :return:
    '''
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
    rospy.spin()

if __name__ == '__main__':
    # load listener
    print("start working...")
    # testing the display GUI
    # for i in range(100):
    #     displayGUI(s1_image, current_image, "sssss deltaRts    " + str(i))
    listener()
