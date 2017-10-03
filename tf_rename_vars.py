"""
 Renames variables for the adversarially trained InceptionV3.
 This is so they can co-exist with the original InceptionV3 variables.

 References: 
   https://stackoverflow.com/questions/37086268/rename-variable-scope-of-saved-model-in-tensorflow/41829414
"""


import sys

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
slim = tf.contrib.slim


BATCH_SHAPE = [16, 299, 299, 3]
NUM_CLASSES = 1001



def dry_run_load(checkpoint_file, scope):
    with tf.Graph().as_default():
      x_input = tf.placeholder(tf.float32, shape=BATCH_SHAPE)
      x_adv = x_input

      with slim.arg_scope(inception.inception_v3_arg_scope()):
        inception.inception_v3(x_input, num_classes=NUM_CLASSES, is_training=False, scope=scope)

      with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(x_adv, num_classes=NUM_CLASSES, is_training=False, reuse=True, scope=scope)

      saver = tf.train.Saver(slim.get_model_variables(scope=scope))
      with tf.Session().as_default() as sess:
        saver.restore(sess, checkpoint_file)

        tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        print(len(tf_vars))
        print(tf_vars[0])



def rename(checkpoint_old, scope_old, checkpoint_new, scope_new):
  with tf.Graph().as_default():
    x_input = tf.placeholder(tf.float32, shape=BATCH_SHAPE)

    with tf.Session().as_default() as sess:
      tf_vars = tf.contrib.framework.list_variables(checkpoint_old)

      new_vars = []
      for name, shape in tf_vars:
        v = tf.contrib.framework.load_variable(checkpoint_old, name)
        print(v.shape) # TEMP
        new_vars.append(tf.Variable(v, name=name.replace(scope_old, scope_new)))

      print('example new variable:', new_vars[0])

      saver = tf.train.Saver(new_vars)
      sess.run(tf.global_variables_initializer())
      saver.save(sess, checkpoint_new)
      print('[info]: saved %d variables' % len(new_vars))



if __name__ == '__main__':
  # path to the original weights.
  # this is the directory + 'adv_inception_v3.ckpt'
  checkpoint_old = '/home/pekalmj1/Documents/cleverhans/examples/nips17_adversarial_competition/sample_defenses/adv_inception_v3/adv_inception_v3.ckpt'
  checkpoint_new = './Weights/InceptionV3-adv.ckpt'

  # Note: it would seem that using "InceptionV3" as the prefix for the new scope may 
  #       cause problems for TF.  WTF.  Anyway, Use a different prefix.
  scope_old = 'InceptionV3'
  scope_new = 'Adv-InceptionV3'


  # make sure we can load the original weights
  dry_run_load(checkpoint_old, scope_old)

  # do it
  rename(checkpoint_old, scope_old, checkpoint_new, scope_new)

  # see if we can load the new weights
  dry_run_load(checkpoint_new, scope_new)



