# InMaxEnt-IRL
Codes and testing tools for Incremental Maximum Entropy Inverse Reinforcement Learning.

# How to use
+ pyTorch_code: code implementation using PyTorch. This version is not completed, I will upload full codes later.
+ tf_code: code implementation using TensorFlow. Also, it's not completed. I will upload later.
+ Tools: Codes for generating training samples.
+ ROS: well, it's a little tricky.
 + We need ROS since we are controlling a WAM robot.
 + Also, ROS is for getting USB-Cam image.
 + We know that ROS Kinetic ONLY works in Python 2.7 if using image transport (CvImage); Python 3.0 is not supported.
 + However, PyTorch and Tensorflow commonly use Python 3.7 or higher.
 + As a result, we designed **img2str to convert image to base64 encodings**. And in PyTorch / Tensorflow, we decode again.

# Reference
 J. Jin, L. Petrich, M. Dehghan, Z. Zhang, M. Jagersand, ”Robot eye-hand coordination learning by watching
human demonstrations: a task function approximation approach”, https://arxiv.org/abs/1810.00159
