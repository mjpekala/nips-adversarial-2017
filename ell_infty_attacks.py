""" ELL_INFTY_ATTACKS   Adversarial attacks with an \ell_\infty constraint on the perturbation.

  A key challenge for this competition is the limited time available to generate an attack.
  We hypothesize carefully addressing hyperparameter selection in conjunction with a strong 
  attack (e.g. [car2017]) may be a reasonable first approach for formulating an attack.

  Note NIPS competition calls the constraint on the maximum per-pixel perturbation \epsilon; 
  here, we call this \tau (following the notation of [car2017]).  

  REFERENCES:
    [car2017]  Carlini & Wagner "Towards Evaluating the Robustness of Neural Networks," 2017.
    [liu2017]  Liu, Chen, Liu & Song "Delving into transferable adversarial examples and black-box attacks," ICLR 2017.
    [kin2017]  Kingma & Ba "Adam: A Method for Stochastic Optimization," ICLR 2015.
"""


__author__ = "mike pekala"
__email__ =  "mpekala@umd.edu"
__date__ = "july 2017 (updated sept 2017)"
__license__ = "Apache 2.0"


import sys, os, time, csv
from copy import copy
from functools import partial
from collections import namedtuple
import pdb 

import numpy as np
np.random.seed(1066)
from scipy.misc import imread, imsave, imresize
import tensorflow as tf



AttackOptions = namedtuple('AttackOptions', 
                           ['learn_rate',     # Adam learning rate
                            'tau',            # constraint on perturbation size  |.|_\infty \leq \tau
                            'c',              # trades off the two terms in loss function
                            't_max',          # max time to spend on an AE (seconds)
                            'sigma',          # variance of initial perturbation
                            'n_restarts',     # number of random restarts (if supported by attack)
                           ])



class AdamAttack:
  """ Optimization-based attacks against a network.
  """

  def __init__(self, opts, f_logit, shape, is_targeted=True):
    self.opts = opts

    self._shape = shape # [1, 299, 299, 3]
    self._num_classes = 1001
    self._is_targeted = is_targeted

    self._init_graph(f_logit)


  def _init_graph(self, f_logit):
    shape = self._shape

    #----------------------------------------------------------------------------
    # inputs
    #----------------------------------------------------------------------------
    self.x_orig_ph = tf.placeholder(tf.float32, shape)        # original (unmodified) image in \R^d
    self.x_0_ph = tf.placeholder(tf.float32, shape)           # start point for optimization in \R^d;

    self.y_tgt_ph = tf.placeholder(tf.int32, shape[0])        # target class

    self.tau_ph = tf.placeholder(tf.float32, [])              # the \ell_\infty constraint in [-1, 1] (scalar)
    self.c_ph = tf.placeholder(tf.float32, []);               # the hyperparameter c (scalar)

    w = tf.Variable(np.zeros(shape, dtype=np.float32))        # the perturbation we are optimizing; in \R^d

    # XXX: could re-enable persistent vars if time permits
    x_orig = self.x_orig_ph
    x_0 = self.x_0_ph
    tau = self.tau_ph
    c = self.c_ph


    #----------------------------------------------------------------------------
    # derived variables
    #----------------------------------------------------------------------------
    x_prime = tf.tanh(x_0 + w)                             # AE in (-1,1)^d
    F_x_prime = f_logit(x_prime)                           # F(x_prime) in notation of [car2017]

    nabla_x = tf.gradients(F_x_prime, x_prime)[0]          # UPDATE: to support fast initial condition

    y_tgt = tf.one_hot(self.y_tgt_ph, self._num_classes)
    z_tgt = tf.reduce_sum(y_tgt * F_x_prime)               # response for the target class (recall y_tgt is onehot)
    if self._is_targeted:
        print('[AdamAttack]: using TARGETED loss function')
        # here we make z_tgt greater than largest response from rest of field
        z_other = tf.reduce_max((1.0 - y_tgt) * F_x_prime) # largest non-target response
        loss_label = (z_other - z_tgt)                 
    else:
        print('[AdamAttack]: using NON-TARGETED loss function')
        # here we make z_tgt smaller than mean response from rest of field
        z_other = tf.reduce_mean((1.0 - y_tgt) * F_x_prime)  # mean response from all other classes
        loss_label = (z_tgt - z_other)                

    delta = x_prime - tf.tanh(x_orig)                      # delta in (-1, 1)^d
    excess = tf.maximum(tf.abs(delta) - tau, 0.0)          # (|delta| - tau)^+
    loss_box = tf.reduce_sum(excess)

    loss = c * loss_label + loss_box                       # the overall loss function

    # mjp: evidently we need to initialize variables after instantiating the optimizer.
    #      see also: https://stackoverflow.com/questions/33788989/tensorflow-using-adam-optimizer
    # 
    # NOTE: AdamOptimizer docs suggest a value of 1.0 or 0.1 for epsilon for imagenet.
    #       https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
    #
    #       HOWEVER, this is for training a CNN, not for developing AE!  
    #       A quick empirical study suggests that using epsilon=0.1 is poor for AE 
    #       (at least when the number of iterations is low).
    #
    # Note: it is also not clear that Adam is ideal in the situation when the number of 
    #       iterations is extremely limited.  We can also experiment with other methods.
    #
    start_vars = set(x.name for x in tf.global_variables())
    optimizer = tf.train.AdamOptimizer(learning_rate=self.opts.learn_rate, epsilon=1e-8, beta1=0.9, beta2=0.999)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
    train_op = optimizer.minimize(loss, var_list=[w])   # minimize w.r.t. w

    end_vars = tf.global_variables()
    new_vars = [x for x in end_vars if x.name not in start_vars]
    self._vars_to_initialize = [w] + new_vars

    # save some references that will be needed when actually attacking
    self._train_op = train_op
    self._loss_label = loss_label
    self._loss_box = loss_box
    self._x_prime = x_prime
    self._F_x_prime = F_x_prime



  def attack(self, sess, x_orig, y_tgt):
    # re-initialize variables.  This is needed to reset the learning rate.
    # I'm not sure if this is necessary, but the intent is to re-initialize Adam.
    init = tf.variables_initializer(self._vars_to_initialize)
    #print(optimizer._lr, K.eval(optimizer._lr_t)) # this is only for debugging
    sess.run(init) 

    # put the features into \R^d
    # the multiplication is to avoid computing atanh(-1) or atanh(1)
    x_orig_R = np.arctanh(x_orig*.999999)  # for x_orig in (-1,1)

    # Here we implement random start locations.
    # Only really makes sense with random multiple restarts...
    #
    delta0 = self.opts.sigma * np.random.rand(*x_orig.shape)
    x_start_R = np.clip(x_orig + delta0, -.999999, 0.999999) # stay in (-1,1)
    x_start_R = np.arctanh(x_start_R)

    # upload data to TF runtime
    #
    # NOTE: Can use TF handles for potential performance gain.
    #       However, this may be less acute now that we attack a batch.
    #
    inputs = {self.x_orig_ph : x_orig_R,
              self.x_0_ph : x_start_R,
              self.y_tgt_ph : y_tgt,
              self.tau_ph : self.opts.tau,
              self.c_ph : self.opts.c}


    # UPDATE: trying FGS as initial condition
    #         Quick experimentation suggests that stepping by a full tau is detrimental.
    #         A smaller step seemed minimally helpful at low # of iterations.
    #         For a reasonable time allotment, I would discard this; however, since we
    #         anticipate extremely limited computational resources, we'll keep it for now.
    #
    if self._is_targeted:
      print('[attack]: WARNING: using FGS initial condition...')
      nabla_x = sess.run([self._x_prime], feed_dict=inputs)[0]
      x_0_fgs = x_orig - (self.opts.tau / 3.0) * np.sign(nabla_x)
      x_0_fgs = np.clip(x_0_fgs, -0.99999, 0.99999)
      inputs[self.x_0_ph] = np.arctanh(x_0_fgs)

    # train for as long as time permits
    tic = time.time() 
    n_iters, loss1_all, loss2_all = 0,[],[]

    while (time.time() - tic) < self.opts.t_max: 
      _, loss1, loss2 = sess.run([self._train_op, self._loss_label, self._loss_box], feed_dict=inputs)
      n_iters += 1
      loss1_all.append(loss1)
      loss2_all.append(loss2)

    # grab the end result from TF and return it to the caller
    x_adv, pred = sess.run([self._x_prime, self._F_x_prime], feed_dict=inputs)

    # see how large the perturbation was
    delta = np.max(np.abs(x_adv - x_orig))

    print('[attack]:     ran %d iters in %0.3f sec (delta=%0.4f)' % (n_iters, (time.time() - tic), delta))

    return x_adv, np.array(loss1_all), np.array(loss2_all), pred
