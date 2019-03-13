import tensorflow as tf
from networks import G,D
from loss import wgan_gp,L2_loss
from dataset import Dataset
from scipy.misc import imread,imsave
import utils
import os
import numpy as np
from argparse import ArgumentParser
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def build_parser():
  parser = ArgumentParser()

  parser.add_argument('--batch_size', type=int,help='batch size',default=2)
  parser.add_argument('--thread_num',type=int,help="data loader thread num",default=2)
  parser.add_argument('--gpus', type=int, help='gpu nums',default=1)
  parser.add_argument('--lr', type=float,help='base learning rate',default=1e-5)
  parser.add_argument('--n_critic', type=int, help='netD/netG', default=50)
  parser.add_argument('--epochs', type=int, help='train epochs', default=150)
  parser.add_argument('--test_dir', type=str, help='test images dir', default="../test")

  return parser


# TODO:multi-gpu train
def train(options):
  lam = 10
  alpha = 1000
  log_nums_per_epoch = 20
  net_gradient_clip_value = 1e8
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    dataset = Dataset(options.batch_size, options.thread_num, "../data")
    steps_per_epoch = dataset.record_number // options.batch_size

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),trainable=False)
    learning_rate = tf.train.exponential_decay(options.lr,global_step,steps_per_epoch*5,0.9, staircase=True)
    d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    real_A=tf.placeholder(tf.float32,(None,512,512,3))
    real_B = tf.placeholder(tf.float32, (None, 512, 512, 3))
    gp_weight_1 = tf.placeholder(tf.float32)
    gp_weight_2 = tf.placeholder(tf.float32)

    # d_tower_grads = []
    # g_tower_grads = []
    # placeholders=[]
    for i in range(options.gpus):
      with tf.device("/gpu:0"):
        with tf.variable_scope("G_B",reuse=tf.AUTO_REUSE):
          fake_A=G(real_B,True)                   #G_B(B)
        with tf.variable_scope("G_A",reuse=tf.AUTO_REUSE):
          fake_B=G(real_A,True)                   #G_A(A)
          test_B=G(real_A,False)                  # test
          rec_B = G(fake_A, True, "bn2")          # G_A(G_B(B))

        with tf.variable_scope("G_B",reuse=tf.AUTO_REUSE):
          rec_A = G(fake_B, True,"bn2")           #G_B(G_A(A))


        with tf.variable_scope("D_A",reuse=tf.AUTO_REUSE):
          d_fake_A = D(fake_A, True)
          d_real_A=D(real_A,True)
          gradient_penalty_1=wgan_gp(fake_A,real_A)*gp_weight_1

        with tf.variable_scope("D_B",reuse=tf.AUTO_REUSE):
          d_fake_B = D(fake_B, True)
          d_real_B = D(real_B, True)
          gradient_penalty_2=wgan_gp(fake_B,real_B)*gp_weight_2


        """ keep in mind that whether the score of groundtruth is high or low doesn't matter """
        wd_B = -tf.reduce_mean(d_fake_B) + tf.reduce_mean(d_real_B)
        wd_A = -tf.reduce_mean(d_real_A) + tf.reduce_mean(d_fake_A)
        netD_train_loss = wd_A + wd_B
        d_loss=-netD_train_loss+gradient_penalty_1+gradient_penalty_2

        _g_loss = tf.reduce_mean(d_fake_B) - tf.reduce_mean(d_fake_A)
        cycle_loss=tf.reduce_mean(tf.stack([L2_loss(real_A,rec_A),L2_loss(real_B,rec_B)]))
        I_loss=tf.reduce_mean(tf.stack([L2_loss(real_A,fake_B),L2_loss(real_B,fake_A)]))
        g_loss=-_g_loss+alpha*I_loss+10*alpha*cycle_loss

        """ show these values in train loop"""
        # true and fake data discriminator score
        dd1 = tf.reduce_mean(d_fake_B)
        dd2 = tf.reduce_mean(d_real_B)
        dd3 = tf.reduce_mean(d_real_A)
        dd4 = tf.reduce_mean(d_fake_A)

        # generator discriminator score
        gg1 = tf.reduce_mean(d_fake_B)
        gg2 = tf.reduce_mean(d_fake_A)

        d_var = utils.trainable_variables('discriminator')
        g_var = utils.trainable_variables('generator')

        d_grads=d_opt.compute_gradients(d_loss,d_var)
        d_capped= [(tf.clip_by_value(grad, -net_gradient_clip_value,net_gradient_clip_value), var) for grad, var in d_grads]
        netD_opt=d_opt.apply_gradients(d_capped)

        g_grads = g_opt.compute_gradients(g_loss, g_var)
        g_capped = [(tf.clip_by_value(grad, -net_gradient_clip_value, net_gradient_clip_value), var) for grad, var in g_grads]
        netG_opt = g_opt.apply_gradients(g_capped)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    log_interval_per_epoch = steps_per_epoch // log_nums_per_epoch
    #save_interval_per_epoch = steps_per_epoch//save_nums_per_epoch
    netD_gp_weight_1 = lam
    netD_gp_weight_2 = lam

    for epoch in range(options.epochs):
      print ("epoch {} out of {}".format(epoch+1,options.epochs))
      for step in range(steps_per_epoch):
        print ("step {} out of {}".format(step+1,steps_per_epoch))
        imagesA, imagesB = dataset.batch()
        feed_dict={real_A:imagesA,real_B:imagesB,gp_weight_1:netD_gp_weight_1,gp_weight_2:netD_gp_weight_2}

        # train D
        _,d,d1,d2,d3,d4=sess.run([netD_opt,netD_train_loss,dd1,dd2,dd3,dd4],feed_dict=feed_dict)
        # train G
        if step%options.n_critic==0:
          feed_dict = {real_A: imagesA, real_B: imagesB}
          _,g,g1,g2=sess.run([netG_opt,_g_loss,gg1,gg2],feed_dict=feed_dict)

        if step%log_interval_per_epoch==0:
          print ("d1:{} d2:{} d3:{} d4:{} d:{}".format(d1,d2,d3,d4,d))
          print ("g1:{} g2:{} g:{}".format(g1,g2,g))


      # each epoch save model and test
      checkpoint_path = os.path.join("../checkpoints", "model_{}.ckpt".format(epoch))
      saver.save(sess,checkpoint_path)
      random_img=utils.get_random(options.test_dir)
      show_img=sess.run(test_B,feed_dict={real_A:random_img})
      #imsave("../{}_real.jpg".format(epoch), random_img[0])
      imsave("../{}_fake.jpg".format(epoch), show_img[0])
      imsave("../{}_fake_clip.jpg".format(epoch), np.clip(show_img[0],0,1))


def main():
  parser = build_parser()
  options = parser.parse_args()
  train(options)


if __name__ == '__main__':
  main()




