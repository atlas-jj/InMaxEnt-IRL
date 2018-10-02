import rospy, roslib, re, sys
sys.path.insert(0,'../')
sys.path.insert(0,'/home/chris/fc/pyTorch_code/InMaxEnt_IRL_Pytorch')
import numpy as np
import base64
import t_utils
import G_model
import G2_model
import R_model_nn
import torch
import copy
from qt_gui.plugin import Plugin
from python_qt_binding.QtCore import Qt, QTimer
from rqt_gui.main import Main
from std_msgs.msg import String

from geometry_msgs.msg import PoseStamped
from python_qt_binding.QtCore import *
from python_qt_binding.QtGui import *
from python_qt_binding.QtWidgets import *
from io import BytesIO
from PIL import Image, ImageDraw, ImageChops
from torchvision import datasets, models, transforms
from colorama import Fore, Back, Style, init as color_init

color_init()


class IRLGuiWidget(QWidget):
#############################################################################################################
# INITIALIZATION METHODS
#############################################################################################################
    def __init__(self):
        super(IRLGuiWidget, self).__init__()
        self.task_name = "blocks3"
        self.setWindowTitle("InMaxEnt_IRL console")

        self.pub = rospy.Publisher('/dRt', String, queue_size=1000)

        self.current_image = Image.open('empty.jpeg')
        self.s1_image = Image.open('empty.jpeg')
        self.sub_image = Image.open('empty.jpeg')

        self.calc_reward_real_time = False  # use the current img as s2 to calc dRt real_time?
        self.update_s1_as_current_auto = False  # automatically set current as s1_image, after calc the deltaRt?

        self.rt_str = "initialized"
        self.inner_count = 0
        self.input_size = 240
        sampling_size = 200
        conf_level = 0.999

        self.current_position = [0, 0, 0]
        self.ref_position = [0, 0, 0]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.task_dim = 3
        self.R = R_model_nn.R_model_NN(self.task_dim, self.device, _sampling_size=sampling_size, _conf_level=conf_level, _lr=0.001)
        self.R.load_weights(torch.load('../../results/R_params0913_plug2_p12345_taskdim3'))
        print('R params loaded!')
        self.R = self.R.to(self.device)
        print(self.R)

        self.logdatas = []

        torch.backends.cudnn.benchmark = True

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        # input("Press Enter to continue...")

        # setup subscribers and publishers
        rospy.Subscriber("/img_str", String, self.callback_getImg)
        rospy.Subscriber("/dRt_CMD", String, self.callback_exe)

        rospy.Subscriber('/zeus/wam/pose', PoseStamped, self.pose_callback, queue_size=1)

        self.main_layout = self.menu_layout()
        self.setLayout(self.main_layout)
        print('inilized!')

#############################################################################################################
# WIDGET
#############################################################################################################
    def button1_function(self):
        self.ref_position = copy.deepcopy(self.current_position)
        self.process_CMD("0")

    def button2_function(self):
        self.process_CMD("1")

    def button3_function(self):
        if self.button3.text() is "Enable Realtime calc":
            self.calc_reward_real_time = True
            self.button3.setText("Disable Realtime calc")
        else:
            self.calc_reward_real_time = False
            self.button3.setText("Enable Realtime calc")


    def button4_function(self):
        self.process_CMD("1")

    def menu_layout(self):
        l = QVBoxLayout()
        l_view_txt = QHBoxLayout()
        l_view = QHBoxLayout()
        l_view2_txt = QHBoxLayout()
        l_view2 = QHBoxLayout()
        l_view3_txt = QHBoxLayout()
        l_view3 = QHBoxLayout()
        button_layout = QHBoxLayout()

        title = QLabel()
        title.setText("InMaxEnt_IRL Console")
        title.setWordWrap(True)
        title.setStyleSheet("font-size: 32px")
        title.setAlignment(Qt.AlignCenter)
        title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        l1 = QLabel()
        l1.setText("Live Cam")
        l1.setAlignment(Qt.AlignCenter)
        l1.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        l2 = QLabel()
        l2.setText("Ref state")
        l2.setAlignment(Qt.AlignCenter)
        l2.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        l_view_txt.addWidget(l1)
        l_view_txt.addWidget(l2)
        l3 = QLabel()
        l3.setText("delta state")
        l3.setAlignment(Qt.AlignCenter)
        l3.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        l4 = QLabel()
        l4.setText("next state")
        l4.setAlignment(Qt.AlignCenter)
        l4.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        l_view2_txt.addWidget(l3)
        l_view2_txt.addWidget(l4)

        l5 = QLabel()
        l5.setText("differential reward:")
        l5.setAlignment(Qt.AlignCenter)
        l5.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        l_view3_txt.addWidget(l5)

        button1 = QPushButton("Set reference")
        button1.clicked.connect(self.button1_function)
        button1.setStyleSheet("background-color: rgba(16, 123, 227, 90%); selection-background-color: rgba(16, 123, 227, 80%); font-size: 32px")
        button1.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        button2 = QPushButton("calc dRt")
        button2.clicked.connect(self.button2_function)
        button2.setStyleSheet("background-color: rgba(16, 123, 227, 90%); selection-background-color: rgba(16, 123, 227, 80%); font-size: 32px")
        button2.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        self.button3 = QPushButton("Enable Realtime calc")
        self.button3.clicked.connect(self.button3_function)
        self.button3.setStyleSheet(
            "background-color: rgba(16, 123, 227, 90%); selection-background-color: rgba(16, 123, 227, 80%); font-size: 32px")
        self.button3.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        self.button4 = QPushButton("Enable auto set ref frame")
        self.button4.clicked.connect(self.button4_function)
        self.button4.setStyleSheet(
            "background-color: rgba(16, 123, 227, 90%); selection-background-color: rgba(16, 123, 227, 80%); font-size: 32px")
        self.button4.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        self._view = QLabel()
        self._view.setAlignment(Qt.AlignCenter)
        self._view.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._view.setFixedSize(280, 280)
        self._view.setFrameShape(2)
        self._view.setFrameShadow(48)
        self._view.setLineWidth(3)
        self._view.setMidLineWidth(3)

        self._view2 = QLabel()
        self._view2.setAlignment(Qt.AlignCenter)
        self._view2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._view2.setFixedSize(280, 280)
        self._view2.setFrameShape(2)
        self._view2.setFrameShadow(48)
        self._view2.setLineWidth(3)
        self._view2.setMidLineWidth(3)

        self._view3 = QLabel()
        self._view3.setAlignment(Qt.AlignCenter)
        self._view3.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._view3.setFixedSize(280, 280)
        self._view3.setFrameShape(2)
        self._view3.setFrameShadow(48)
        self._view3.setLineWidth(3)
        self._view3.setMidLineWidth(3)

        self._view3_current = QLabel()
        self._view3_current.setAlignment(Qt.AlignCenter)
        self._view3_current.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._view3_current.setFixedSize(280, 280)
        self._view3_current.setFrameShape(2)
        self._view3_current.setFrameShadow(48)
        self._view3_current.setLineWidth(3)
        self._view3_current.setMidLineWidth(3)

        self._view4 = QLabel()
        self._view4.setText("test")
        # self._view4.setWordWrap(True)
        self._view4.setStyleSheet("font-size: 32px; font-weight: bold;")
        self._view4.setAlignment(Qt.AlignCenter)
        self._view4.setFixedSize(600, 80)
        self._view4.setFrameShape(2)
        self._view4.setFrameShadow(48)
        self._view4.setLineWidth(3)
        self._view4.setMidLineWidth(3)
        self._view4.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        button_layout.addWidget(button1)
        button_layout.addWidget(button2)
        # button_layout.addWidget(self.button3)
        # button_layout.addWidget(self.button4)
        l_view.addWidget(self._view3_current)  # show current image
        l_view.addWidget(self._view)  # show ref image
        l_view2.addWidget(self._view3)  # show sub image
        l_view2.addWidget(self._view2)  # show target image.
        l_view3.addWidget(self._view4)  # show delta R string

        l.addWidget(title)
        l.addLayout(l_view_txt)
        l.addLayout(l_view)
        l.addLayout(l_view2_txt)
        l.addLayout(l_view2)
        l.addLayout(l_view3_txt)
        l.addLayout(l_view3)
        l.addLayout(button_layout)
        return l

#############################################################################################################
# CALLBACK METHODS
#############################################################################################################
    def callback_getImg(self, data):
        '''
            subscribe topics publishing image base64 string, without header.
            :param data:
            :return:
            '''
        # rospy.loginfo(rospy.get_caller_id() + "img_str %s", data.data[0:10])
        if data.data is "" or data.data is "h":
            print('img str base64 encode invalid!')
            return
        # process the image data, and publish an error msg.
        self.current_image = Image.open(BytesIO(base64.b64decode(data.data)))
        p = self.convert_img_pil(self.current_image)
        self._view3_current.setPixmap(p)

        #  display current image
        if self.calc_reward_real_time:
            self.get_d_Rt(1)

    def callback_exe(self, data):
        '''
        subscribe to /dRt_CMD, when got 0: mark s1, when got 1: using current_image as s2, s2-s1, and get the deltaRewards
        :param data: string 0/1
        :return:
        '''
        self.process_CMD(data.data)

    def pose_callback(self, data):
        self.current_position[0] = data.pose.position.x * 1000
        self.current_position[1] = data.pose.position.y * 1000
        self.current_position[2] = data.pose.position.z * 1000
        # self._view4.setText(str(self.current_position[0]) + "," + str(self.current_position[1]) + "," + str(self.current_position[2]))
        # self._view4.update()
        # print(self.current_position)


#############################################################################################################
# utils
#############################################################################################################
    def convert_img(self, data):
        try:
            img = Image.open(BytesIO(base64.b64decode(data)))
        except:
            print ("1")
        try:
            frame = np.asarray(img)
        except:
            print ("2")
        try:
            qimg = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_Indexed8)
        except:
            print ("3")
        pixmap = QPixmap.fromImage(qimg)
        return pixmap

    def convert_img_pil(self, img):
        try:
            frame = np.asarray(img)
        except:
            print ("2")
        try:
            qimg = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_Indexed8)
        except:
            print ("3")
        pixmap = QPixmap.fromImage(qimg)
        return pixmap

    def conver2str(self, np_vector):
        rs = ""
        rs_inv = ""
        for i in range(np_vector.shape[0]):
            rs += (str(0 - np_vector[i]))[0:5] + " "
            rs_inv +=  (str(np_vector[i]))[0:5] + " "
        return rs[0:(len(rs) - 1)], rs_inv[0:(len(rs_inv) - 1)]

    def process_CMD(self, cmd):
        if cmd is "0":  # mark s1_image, it's 0 only at the initial stage, when put in practice
            self.s1_image = copy.deepcopy(self.current_image)
            # update view_0
            self._view.setPixmap(self.convert_img_pil(self.s1_image))
        elif cmd is "1":  # mark current image as s2_image, calculate dS, and output dRt
            print('get_drt')
            self.get_d_Rt(1)

    def get_d_Rt(self, interval_count):
        '''
        calc one delta Rt, and update s1_image, rt_str, and also the display GUI
        :return:
        '''
        self.inner_count += 1
        if self.inner_count >= interval_count:
            inner_count = 0
            samples = torch.zeros(1, 1, self.input_size, self.input_size)
            self.sub_image = ImageChops.subtract_modulo(self.current_image, self.s1_image)
            # now transform to py tensor and feed to the network
            samples[0] = self.transform(self.sub_image)
            delta_Rewards = self.R.forward(samples.to(self.device), False)
            # delta_rewards[0] to string
            rt_str, rt_inv_str = self.conver2str(delta_Rewards[0].detach().to('cpu').numpy())
            print(rt_str)
            # now publish this delta_reward
            self.pub.publish(rt_str)
            # display results
            self._view3.setPixmap(self.convert_img_pil(self.sub_image))
            self._view2.setPixmap(self.convert_img_pil(self.current_image))
            self._view4.setText(rt_inv_str)
            # self._view4.setText(rt_str)
            self._view4.update()
            this_v = [self.ref_position[0], self.ref_position[1], self.ref_position[2], self.current_position[0],
                      self.current_position[1], self.current_position[2], float(delta_Rewards[0][0]), float(delta_Rewards[0][1]), float(delta_Rewards[0][2])]
            self.logdatas.append(this_v)
            np.save(self.task_name+'_reward_space', self.logdatas)
            # now re mark current image as s1 image
            if self.update_s1_as_current_auto is True:
                self.s1_image =copy.deepcopy(self.current_image)


#############################################################################################################
# EVENT METHODS
#############################################################################################################
    def keyPressEvent(self, QKeyEvent):
        key = QKeyEvent.key()
        if int(key) == 82: # r
            self.reset()
        elif int(key) == 68: # d
            self.debug()
        elif int(key) == 81: # q
            sys.exit()
        else:
            print("Unknown key option: " + str(key))


class IRLGuiPlugin(Plugin):
    def __init__(self, context):
        super(IRLGuiPlugin, self).__init__(context)
        if context.serial_number() > 1:
            raise RuntimeError("You may not run more than one instance of visual_servoing_gui.")
        self.setObjectName("Visual Servoing Plugin")
        self._widget = IRLGuiWidget()
        context.add_widget(self._widget)

if __name__ == '__main__':
    main = Main()
    print('run rqt now')
    sys.exit(main.main(sys.argv, standalone='irl_gui.irl_gui.IRLGuiPlugin'))
