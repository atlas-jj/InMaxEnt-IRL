import numpy as np
import t_utils
import G_model
import G2_model
import R_model_nn
import torch
import gc
import time
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style, init as color_init

color_init()

class InMaxEnt_IRL:
    def __init__(self):
        """
          Incremental Maximum Entropy IRL
          the main entrance
          inputs:

          returns

          """
        #self.feat_map = feat_map
        self.sampling_size = 480
        self.conf_level = 0.999

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        task_dim = 4
        self.R = R_model_nn.R_model_NN(task_dim, self.device, _lr=0.001)
        self.R = self.R.to(self.device)
        print(self.R)
        self.G = G_model.G_model(task_dim, self.device, _lr=0.001)
        self.G = self.G.to(self.device)
        print(self.G)


    def train(self):
        """
        perform training
        """
	sampling_size = 200
	conf_level = 0.997  #  0.999
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	task_dim = 3
	self.R = R_model_nn.R_model_NN(task_dim, device, _sampling_size=sampling_size, _conf_level=conf_level, _lr=0.001)
	# R.load_weights(torch.load('generate_dataset/output4NNmodels_torchV2'))
	# R.load_weights(torch.load('generate_dataset/output4NNmodels_torchV2'))
	R = R.to(device)
	print(R)
	#G = G_model.G_model(task_dim, device, _sampling_size=sampling_size, _conf_level=conf_level, _lr=0.0005)
	#G.load_weights(torch.load('results/G_params_0229_G'))
	#G = G.to(device)
	#print(G)
	torch.backends.cudnn.benchmark = True
	# load pretrained R model, otherwise, train R model firstly
	inputs1 = torch.load('generate_dataset/plug2_p1_delta_samples_4')
	inputs2 = torch.load('generate_dataset/plug2_p2_delta_samples_4')
	inputs3 = torch.load('generate_dataset/plug2_p3_delta_samples_4')
	inputs4 = torch.load('generate_dataset/plug2_p4_delta_samples_4')
	inputs5 = torch.load('generate_dataset/plug2_p5_delta_samples_4')
	#
	inputs1_inv = torch.load('generate_dataset/plug2_p1_delta_samples_inverse_4')
	inputs2_inv = torch.load('generate_dataset/plug2_p2_delta_samples_inverse_4')
	inputs3_inv = torch.load('generate_dataset/plug2_p3_delta_samples_inverse_4')
	inputs4_inv = torch.load('generate_dataset/plug2_p4_delta_samples_inverse_4')
	inputs5_inv = torch.load('generate_dataset/plug2_p5_delta_samples_inverse_4')

	inputs = torch.cat((inputs1, inputs2, inputs3, inputs4, inputs5), 0)
	inputs_inv = torch.cat((inputs1_inv, inputs2_inv, inputs3_inv, inputs4_inv, inputs5_inv), 0)

	epoch = 1000
	input("Press Enter to continue...")

	epoch = 3000
	R_cost_array=[]
	G_cost_array=[]
	Zts_array = []
	Zts_inv_array = []

	for i in range(epoch):
	    # initialize R
	    #R.train_step_test_inverse_combined(inputs.detach().to(device), inputs_inv.detach().to(device), i)
	    # cost = R.train_0730_model2(inputs.detach(), inputs_inv.detach(), i) # success!
	    cost = self.R.train_0730_model(inputs.detach(), inputs_inv.detach(), i)
	    # if cost <= -1.99999:
	    #    break
	    R_cost_array.append(cost)

	    # print('sorry, system fell asleep 30s to cool down our expensive GPU...')
	    # time.sleep(20) # freez the GPU
	    # input("Press Enter to continue...")

	input("Press Enter to continue...")

	torch.cuda.empty_cache()

	# train G
	# epoch = 1000

	# G.train_step_sub(inputs, R, 1000)


    def predict(self):
        """
        perform prediction
        input:
          delta_Ss    NxD matrix - the features for each state
        output:
          reward
        """
        #  return self.nn_r.feedforward(delta_Ss)


my = InMaxEnt_IRL()
my.train()

