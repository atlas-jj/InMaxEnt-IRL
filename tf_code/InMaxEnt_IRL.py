import numpy as np
import tensorflow as tf
import img_utils
import tf_utils
import utils
import models as nn_models
import math
from sklearn.preprocessing import Normalizer
import dataloader as dtl
import pdb
from sklearn.utils import shuffle
#previous deep irl algorithm, the traj is defined as a collection of prev_state, action and next_state
# traj, e.g., trainset[0]
class InMaxEnt_IRL:
    def __init__(self, traj , task_dim, de_lambda, conf_level, lr, n_iters, method, sampling_size, hn_1, hn_2):
        """
          Incremental Maximum Entropy IRL
          the main entrance
          inputs:
            # feat_map    NxD matrix - the features for each state
            traj        a sequence of demonstrated states, NxD matrix - the features for each state
            task_dim    int, task dimension, the reward should be a vector of the same dimension
            de_lambda   float, in [0,1), demonstrator's self-assessment factor, lambda=bad_actions / total_actions, the smaller the better.
            conf_level  float, in [0,1], demonstrator's self-assessment factor, the average how confident when you select the best action in each state.
            lr          float - learning rate
            n_iters     int - number of optimization steps
            method      "fc", use fully connected layers; "conv", use convolution layers
          returns
            rewards     Nx1 vector - recoverred state rewards
          """
        #self.feat_map = feat_map
        self.traj = traj
        self.de_lambda = de_lambda
        self.conf_level = conf_level
        self.lr = lr
        self.n_iters = n_iters
        self.task_dim = task_dim
        l2=0.1
        self.sess = tf.Session()
        self.nn_r = nn_models.InMaxEntIRL_FC(sess, traj.shape[1], lr, hn_1, hn_2, 'chris_fc', task_dim, l2) #l2 regularization weight
        self.sampling_size = sampling_size
        self.sess.run(tf.global_variables_initializer())

        #if method == "conv":
            #self.nn_r = nn_models.InMaxEntIRL_Conv(feat_map.shape[1], lr, 64, 64) # use fully connected layers, two hidden layers.

    def train(self, y_labels):
        """
        perform training
        """
        # training
        for iteration in range(self.n_iters):
            if iteration % (self.n_iters/10) == 0:
                print ('iteration: {}'.format(iteration))
            #theta = self.nn_r.get_theta()
            grad_theta_example = self.nn_r.backprop(np.zeros(self.task_dim), np.ones(self.traj.shape[1])) # get the grad_theta structure
            grad_theta = utils.get_zero_grad_theta_value(grad_theta_example) ###88888888888888 return 0 gradient theta

            #code only for test
            delta_rewards = self.nn_r.feedforward(self.traj)
            #pdb.set_trace()
            # compute current crossEntropyError
            error = utils.crossEntropyError(y_labels,delta_rewards)
            print('NN, iteration: ' + str(iteration)+', mean cross entropy: ' + str(error))
            #end of code only for test
            nabla_delta_reward = nabla_crossEnt(y_labels, delta_rewards)
            grad_theta222 = nn_r.backprop(nabla_delta_reward, self.traj)
            print("grad theta, using patch: " )
            print(grad_theta222)

            for t in range(1):#self.traj.shape[0]
                delta_St = self.traj[t]
                delta_St = utils.convert2Row(delta_St)
                delta_Rt = self.nn_r.feedforward(delta_St) ###888888888
                #print(delta_Rt.shape)

                d_Rt = utils.dRt_fn(delta_Rt)
                grad_deltaRt = utils.dRt_Gradient(delta_Rt)

                grad_theta2 = self.nn_r.backprop(grad_deltaRt, delta_St) ##88grad_deltaRt as error sig, delta_St as input x

                #delta_Rtj_list = utils.sample_deltaRtj(d_Rt, self.de_lambda, self.conf_level, self.sampling_size, self.task_dim)
                #Zt = utils.comput_Zt(delta_Rtj_list)

                grad_theta1 = utils.get_zero_grad_theta_value(grad_theta_example)

                # for j in range(self.sampling_size):
                #     delta_Rtj = delta_Rtj_list[j]
                #     dRtj = utils.dRt_fn(delta_Rtj)
                #     grad_deltaRtj = utils.dRt_Gradient(delta_Rtj) * (math.exp(dRtj)) / Zt
                #     grad_theta1 = utils.add_grad_theta(grad_theta1 , self.nn_r.backprop(grad_deltaRtj, delta_St)) ####88888888

                # only for test purpose
                grad_theta2 = utils.get_zero_grad_theta_value(grad_theta2)
                nabla_delta_Rt =utils.nabla_crossEntSingle(y_labels[t], delta_Rt[0][0])

                nabla_delta_Rt = np.ones((1,1))*nabla_delta_Rt
                grad_theta1 = self.nn_r.backprop(nabla_delta_Rt, delta_St)
                # end of only for test purpose
                #pdb.set_trace()
                grad_theta_temp = utils.minus_grad_theta(grad_theta1, grad_theta2)
                #pdb.set_trace()
                grad_theta = utils.add_grad_theta(grad_theta, grad_theta_temp)
                #pdb.set_trace()
            # compute new theta
            # apply gradient to this theta
            #print("grad_norm : " + str(grad_norms))
            grad_theta = utils.scale_grad_theta(grad_theta, 1/self.traj.shape[0])
            print("grad theta, iterative final " )
            print(grad_theta)
            #self.nn_r.apply_grads(grad_theta)
    def predict(self, delta_Ss):
        """
        perform prediction
        input:
          delta_Ss    NxD matrix - the features for each state
        output:
          reward
        """
        return self.nn_r.feedforward(delta_Ss)
