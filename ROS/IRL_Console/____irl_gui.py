import rospy, roslib, re, sys
sys.path.insert(0,'../')
sys.path.insert(0,'/home/chris/catkin_ws/src/IRL_GUIs/irl_gui/src')
import numpy as np
import base64
from qt_gui.plugin import Plugin
from python_qt_binding.QtCore import Qt, QTimer
from rqt_gui.main import Main
from std_msgs.msg import String
from python_qt_binding.QtCore import *
from python_qt_binding.QtGui import *
from python_qt_binding.QtWidgets import *
from io import BytesIO
from PIL import Image, ImageDraw, ImageChops

class IRLGuiWidget(QWidget):
#############################################################################################################
# INITIALIZATION METHODS
#############################################################################################################
    def __init__(self):
        print('1!')
        super(IRLGuiWidget, self).__init__()
        self.setWindowTitle("IRL Widget")
        # setup subscribers and publishers
        print('2!')
        rospy.Subscriber("/im1", String, self.callback)
        rospy.Subscriber("/img_str", String, self.callback2)
        rospy.Subscriber("/im3", String, self.callback3)
        rospy.Subscriber("/im_info", String, self.callback4)
        print('3!')
        self.pub = rospy.Publisher("/dRt_CMD", String, queue_size = 1)
        self.main_layout = self.menu_layout()
        self.setLayout(self.main_layout)
        print('inilized!')

#############################################################################################################
# WIDGET
#############################################################################################################
    def button1_function(self):
        print ("sending 0")
        self.pub.publish("0")

    def button2_function(self):
        print ("sending 1")
        self.pub.publish("1")


    def menu_layout(self):
        l = QVBoxLayout()
        l_view = QHBoxLayout()
        l_view2 = QHBoxLayout()
        button_layout = QHBoxLayout()

        title = QLabel()
        title.setText("IRL")
        title.setWordWrap(True)
        title.setStyleSheet("font-size: 32px")
        title.setAlignment(Qt.AlignCenter)
        title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)


        button1 = QPushButton("Send 0")
        button1.clicked.connect(self.button1_function)
        button1.setStyleSheet("background-color: rgba(16, 123, 227, 90%); selection-background-color: rgba(16, 123, 227, 80%); font-size: 32px")
        button1.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        button2 = QPushButton("Send 1")
        button2.clicked.connect(self.button2_function)
        button2.setStyleSheet("background-color: rgba(16, 123, 227, 90%); selection-background-color: rgba(16, 123, 227, 80%); font-size: 32px")
        button2.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

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

        self._view4 = QLabel()
        self._view4.setText("test")
        self._view4.setWordWrap(True)
        self._view4.setStyleSheet("font-size: 15px; font-weight: bold;")
        self._view4.setAlignment(Qt.AlignCenter)
        self._view4.setFixedSize(280, 280)
        self._view4.setFrameShape(2)
        self._view4.setFrameShadow(48)
        self._view4.setLineWidth(3)
        self._view4.setMidLineWidth(3)
        self._view4.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        button_layout.addWidget(button1)
        button_layout.addWidget(button2)
        l_view.addWidget(self._view)
        l_view.addWidget(self._view2)
        l_view2.addWidget(self._view3)
        l_view2.addWidget(self._view4)

        l.addWidget(title)
        l.addLayout(l_view)
        l.addLayout(l_view2)
        l.addLayout(button_layout)
        return l

#############################################################################################################
# CALLBACK METHODS
#############################################################################################################
    def callback(self, data):
        s = data.data
        p = self.convert_img(s)
        self._view.setPixmap(p)

    def callback2(self, data):
        s = data.data
        p = self.convert_img(s)
        self._view2.setPixmap(p)

    def callback3(self, data):
        s = data.data
        p = self.convert_img(s)
        self._view3.setPixmap(p)

    def callback4(self, data):
    	msg = str(data.data)
    	self._view4.setText(msg)

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
    sys.exit(main.main(sys.argv, standalone='IRLGuiPlugin'))
