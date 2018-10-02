#!/usr/bin/env python
from __future__ import print_function

import roslib
# roslib.load_manifest('img2str')
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import base64
import cStringIO
from PIL import Image as Image2

class Img2Str:

  	def __init__(self):
		self.bridge = CvBridge()
		self.image_pub = rospy.Publisher("/img_str", String, queue_size = 1000)
		self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback)

  	def callback(self, data):
		try:
		  	cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
		  	print(e)
		self.crop_image(cv_image)

	def crop_image(self, img):
		sz = 240
		(rows,cols,channels) = img.shape
		mask_top_left =  [849,169]  # [460,0]
		mask_bottom_right = [1429,749]  #[1390,930]
		# mask_top_left = [755,18]
		# mask_bottom_right = [1361,624]
		# h = (rows - sz) / 2
		# w = (cols - sz) / 2
		roi_t = img[mask_top_left[1]:mask_bottom_right[1], mask_top_left[0]:mask_bottom_right[0]]
		roi_t = cv2.GaussianBlur(roi_t,(3,3),0)
		roi = cv2.resize(roi_t, (sz, sz))
		# convert to gray scale
		roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		# conver to pillow image
		pil_img = Image2.fromarray(roi_gray)
		buffer = cStringIO.StringIO()
		# conver to base64 encodes
		pil_img.save(buffer, format="JPEG")
		img_str = base64.b64encode(buffer.getvalue())
		# encoded_roi = base64.encodestring(np.ascontiguousarray(roi))


		cv2.imshow("ROI", roi_t)
		cv2.waitKey(3)
		# self.image_pub.publish(encoded_roi)
		self.image_pub.publish(img_str)

def main(args):
  rospy.init_node('Img2Str', anonymous=True)
  ic = Img2Str()
  try:
	rospy.spin()
  except KeyboardInterrupt:
	print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
