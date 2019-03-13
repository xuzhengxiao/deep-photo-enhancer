import tensorflow as tf
import random
from scipy.misc import imread,imsave,imresize
import os
import numpy as np

def get_random(test_dir):
  test_images = os.listdir(test_dir)
  img_path = random.choice(test_images)
  img = imread(os.path.join(test_dir,img_path))
  img = imresize(img, (512, 512)) / 255.0
  img = img[np.newaxis, ...]
  return img[...,:3]

def tensors_filter(tensors, filters, combine_type='or'):
  assert isinstance(tensors, (list, tuple)), '`tensors` shoule be a list or tuple!'
  assert isinstance(filters, (str, list, tuple)), \
    '`filters` should be a string or a list(tuple) of strings!'
  assert combine_type == 'or' or combine_type == 'and', "`combine_type` should be 'or' or 'and'!"

  if isinstance(filters, str):
    filters = [filters]

  f_tens = []
  for ten in tensors:
    if combine_type == 'or':
      for filt in filters:
        if filt in ten.name:
          f_tens.append(ten)
          break
    elif combine_type == 'and':
      all_pass = True
      for filt in filters:
        if filt not in ten.name:
          all_pass = False
          break
      if all_pass:
        f_tens.append(ten)
  return f_tens


def trainable_variables(filters=None, combine_type='or'):
  t_var = tf.trainable_variables()
  if filters is None:
    return t_var
  else:
    return tensors_filter(t_var, filters, combine_type)
