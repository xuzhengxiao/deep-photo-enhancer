import tensorflow as tf
from networks import G
from scipy.misc import imread,imresize,imsave
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def random_pad_to_size(img, size,pad_symmetric=True, use_random=True):

  s0 = size - img.shape[0]
  s1 = size - img.shape[1]

  if use_random:
    b0 = np.random.randint(0, s0 + 1)
    b1 = np.random.randint(0, s1 + 1)
  else:
    b0 = 0
    b1 = 0
  a0 = s0 - b0
  a1 = s1 - b1
  if pad_symmetric:
    img = np.pad(img, ((b0, a0), (b1, a1), (0, 0)), 'symmetric')
  else:
    img = np.pad(img, ((b0, a0), (b1, a1), (0, 0)), 'constant')
  return img,[b0, img.shape[0] - a0, b1, img.shape[1] - a1]

def test(img_path):
  img=imread(img_path)
  h,w,_=img.shape
  rec=None
  if h>512 or w>512:
    img=imresize(img,(512,512))
  else:
    img,rec=random_pad_to_size(img,512)
  img=img[np.newaxis,...][...,:3]/255.0
  model_path="../checkpoints/model_40.ckpt"
  real_A = tf.placeholder(tf.float32, (None, 512, 512, 3))
  with tf.variable_scope("G_A"):
    fake_B = G(real_A, False)
  saver = tf.train.Saver()
  with tf.Session() as sess:
    saver.restore(sess, model_path)
    feed_dict={real_A:img}
    _img=sess.run(fake_B,feed_dict=feed_dict)
  img=_img[0]
  if rec!=None:
    img=img[rec[0]:rec[1],rec[2]:rec[3]]
  imsave("../generate.jpg",np.clip(img,0,1))


if __name__ == '__main__':
  test("../2.jpg")
