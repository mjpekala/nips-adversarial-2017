"""  Code for tensorflow classifiers.
"""

__author__ = "mjp"
__date__ = "september, 2017"


import time, os

import numpy as np

from scipy.misc import imread, imsave

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import resnet_v2

import ens_adv_inception_resnet_v2.inception_resnet_v2 as ir

slim = tf.contrib.slim



#-------------------------------------------------------------------------------

def _load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]

  all_files = tf.gfile.Glob(os.path.join(input_dir, '*.png'))

  #for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
  for filepath in all_files:
    with tf.gfile.Open(filepath, mode='rb') as f:
      image = imread(f, mode='RGB').astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0

  if idx > 0:
    yield filenames, images


def smooth_one_hot_predictions(p):
  """Given a vector of predicted class labels p, generates a 'smoothed' one-hot prediction matrix."""
  out = (1./NUM_CLASSES) * np.ones((p.size, NUM_CLASSES), dtype=np.float32)
  for ii in range(p.size):
    out[ii,p[ii]] = 0.9
  return out


def gradient_norm(G):
  """ Computes two norm of the tensor G whose first dimension is assumed to be # of examples. """
  out = G ** 2
  while (out.ndim > 1):
    out = np.sum(out, axis=out.ndim-1)
  return np.sqrt(out)


#-------------------------------------------------------------------------------

class Generic_Network:
    def __init__(self, scope, weights_file, 
                 shape=[16, 299, 299, 3],
                 num_classes=1001,
                 tf_master=''):
      """ Initialize network properties. """

      self._scope = scope
      self._weights_file = weights_file

      self._shape = shape
      self._tf_master = tf_master
      self._num_classes = num_classes

      self._label_smoothing = 0.1
      self._baseline_dir = './Baselines'

      self.name = self._scope


    def _baseline_filename(self):
      return os.path.join(self._baseline_dir, 'STATS_'+self.name+'.npz')


    def __call__(self, *args, **kargs):
      "This is just some convenient shorthand."
      return self.predict_all_images_with_confidence(*args, **kargs)


    def _prediction_graph_output(self, x_input, pre_softmax=False):
      """ Returns a tensorflow variable representing the (scalar) predictions 
          (ie. prediction outputs).
      """
      raise RuntimeError('this method should be overridden!')


    def _dloss_dx(self, x_input, y_hat_input):
      """ Returns a tensorflow variable representing the derivative of the loss w.r.t. the input.
          This is evaluated at (x,y)
      """
      raise RuntimeError('this method should be overridden!')


    def predict_all_images(self, input_dir):
      """ Generates predictions (only) for all images in the input directory.

      If you also want confidence scores, use predict_all_images_with_confidence() instead.
      """
      predictions_all = []
      files_all = []
      tic = time.time()

      with tf.Graph().as_default():

        #
        # All models must take the same input.
        # The output variable depends upon the specific network.
        #
        x_input = tf.placeholder(tf.float32, shape=self._shape)
        y_hat_output = self._prediction_graph_output(x_input)

        saver = tf.train.Saver(slim.get_model_variables(scope=self._scope))
        session_creator = tf.train.ChiefSessionCreator(master=self._tf_master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
          saver.restore(sess, self._weights_file)

          for filenames, images in _load_images(input_dir, self._shape):
            preds = sess.run(y_hat_output, feed_dict={x_input : images})
            preds = preds[0:len(filenames)]  # note: the last mini-batch may have fewer images

            predictions_all.append(preds)
            files_all.extend(filenames)

      print('[%s]: time to predict %d images: %0.2f (sec)' % (self.name, len(files_all), time.time()-tic))

      # concatenate the prediction arrays into one big array 
      predictions_all = np.concatenate(predictions_all)
      return files_all, predictions_all



    def predict_all_images_with_confidence(self, input_dir, save_baseline=False):
      """ Generates class predictions and associated confidence scores.
      """

      #--------------------------------------------------
      # STEP 1 : Initial predictions
      #--------------------------------------------------
      # First, generate predictions on all images.
      files_all, y_hat_all = self.predict_all_images(input_dir)

      # XXX: I don't think we want to use smoothed predictions here...
      #      For this application, we want the "native" prediction?
      #if y_hat_all.ndim == 1:
      #  print('[INFO]:  ** WARNING ** using smoothed predictions!!')
      #  y_hat_all = smooth_one_hot_predictions(y_hat_all)

      #--------------------------------------------------
      # STEP 2 : "Confidence" estimates
      #--------------------------------------------------
      y_hat_minibatch = np.zeros((self._shape[0], self._num_classes), np.float32)
      conf_all = []
      tic = time.time()

      with tf.Graph().as_default():
        # All models must take the same input
        x_input = tf.placeholder(tf.float32, shape=self._shape)
        y_hat_input = tf.placeholder(tf.float32, shape=[self._shape[0], self._num_classes])

        # The gradient calculation depends upon the underlying graph
        nabla_x = self._dloss_dx(x_input, y_hat_input)

        # Evaluate
        saver = tf.train.Saver(slim.get_model_variables(scope=self._scope))
        session_creator = tf.train.ChiefSessionCreator(master=self._tf_master)
        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
          saver.restore(sess, self._weights_file)

          idx = 0
          for filenames, images in _load_images(input_dir, self._shape):
            # grab corresponding subset of predictions
            y_hat_minibatch[0:len(filenames),:] = y_hat_all[idx:(idx+len(filenames)),:]
            idx += len(filenames)

            feed_dict = {x_input : images, y_hat_input : y_hat_minibatch}

            grad = sess.run(nabla_x, feed_dict=feed_dict)
            grad = grad[0:len(filenames),...]  # Note: last batch may contain fewer images
            conf_all.append(gradient_norm(grad))

      print('[%s]: time to generate confidence scores for %d images: %0.2f (sec)' % (self.name, len(files_all), time.time()-tic))

      # reshape into a single tensor 
      conf_all = np.concatenate(conf_all)

      if save_baseline:
        np.savez(self._baseline_filename(), conf=conf_all, preds=np.argmax(y_hat_all, axis=1))
        print('[%s]: saved %d predictions to baseline file' % (self.name, conf_all.size))

      return files_all, y_hat_all, conf_all



    def confidence_quantile(self, v):
      """ Compares confidence scores v to those in the baseline.
      """
      if os.path.exists(self._baseline_file()):
        conf = np.load(self._baseline_file())['conf']
        quantiles = np.zeros((v.size,))
        for ii in range(v.size):
          quantiles[ii] = 1.0 * np.sum(v[ii] > conf) / conf.size
        return quantiles

      else:
        return np.nan


#-------------------------------------------------------------------------------


class InceptionV3(Generic_Network):
    def __init__(self, *args, **kargs):
        Generic_Network.__init__(self, *args, **kargs)


    def _prediction_graph_output(self, x_input, pre_softmax=False):
      with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(x_input, num_classes=self._num_classes, is_training=False, scope=self._scope)

      if pre_softmax:
        return logits
      else:
        # note: predictions seem to be "probabilities" (softmax outputs)
        #return tf.nn.softmax(logits)
        return end_points['Predictions']


    def _dloss_dx(self, x_input, y_hat_input):
      with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(x_input, num_classes=self._num_classes, is_training=False, scope=self._scope)

      #cross_entropy = tf.losses.softmax_cross_entropy(y_hat_input,
      #                                                logits,
      #                                                label_smoothing=0.1,
      #                                                weights=1.0)
      #cross_entropy += tf.losses.softmax_cross_entropy(y_hat_input,
      #                                                 end_points['AuxLogits'],
      #                                                 label_smoothing=0.1,
      #                                                 weights=0.4)

      cross_entropy = tf.losses.softmax_cross_entropy(y_hat_input,
                                                      #logits,
                                                      end_points['Predictions'],
                                                      label_smoothing=self._label_smoothing,
                                                      weights=1.0)

      nabla_x = tf.gradients(cross_entropy, x_input)[0]
      return nabla_x


#-------------------------------------------------------------------------------


class ResnetV2(Generic_Network):
    def __init__(self, *args, **kargs):
        Generic_Network.__init__(self, *args, **kargs)


    def _prediction_graph_output(self, x_input, pre_softmax=False):
      with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, _ = resnet_v2.resnet_v2_101(x_input, num_classes=self._num_classes, is_training=False, scope=self._scope)

      # note: logits have shape [BATCH_SIZE, 1, 1, NUM_CLASSES]
      # Hence the squeeze below.  Assumes BATCH_SIZE > 1.
      logits = tf.squeeze(logits)

      if pre_softmax:
        return logits
      else:
        return tf.nn.softmax(logits)


    def _dloss_dx(self, x_input, y_hat_input):
      with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, _ = resnet_v2.resnet_v2_101(x_input, num_classes=self._num_classes, is_training=False, scope=self._scope)

      cross_entropy = tf.losses.softmax_cross_entropy(y_hat_input,
                                                      tf.squeeze(logits),  # note: squeeze needed here
                                                      label_smoothing=self._label_smoothing,
                                                      weights=1.0)

      nabla_x = tf.gradients(cross_entropy, x_input)[0]

      return nabla_x


#-------------------------------------------------------------------------------


class AdvResnetV2InceptionV3(Generic_Network):
  """ Adversarially trained ensemble of {ResnetV2, InceptionV3} provided by competition organizers.
  """
  def __init__(self, *args, **kargs): 
    Generic_Network.__init__(self, *args, **kargs)


  def _prediction_graph_output(self, x_input, pre_softmax=False):
    with slim.arg_scope(ir.inception_resnet_v2_arg_scope()): 
      logits, _ = ir.inception_resnet_v2(x_input, num_classes=self._num_classes, is_training=False, scope=self._scope)

    if pre_softmax:
      return logits
    else:
      return tf.nn.softmax(logits)


  def _dloss_dx(self, x_input, y_hat_input):
    with slim.arg_scope(ir.inception_resnet_v2_arg_scope()): 
      logits, _ = ir.inception_resnet_v2(x_input, num_classes=self._num_classes, is_training=False, scope=self._scope)

    cross_entropy = tf.losses.softmax_cross_entropy(y_hat_input, 
                                                    logits,
                                                    label_smoothing=self._label_smoothing,
                                                    weights=1.0)
    nabla_x = tf.gradients(cross_entropy, x_input)[0]

    return nabla_x


