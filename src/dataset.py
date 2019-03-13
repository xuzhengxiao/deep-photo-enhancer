from queue import Queue
from threading import Thread
import random
import os
from scipy.misc import imread,imresize
import numpy as np

class Dataset(object):
  def __init__(self,batch_size,thread_num,dataroot):
    self.batch_size = batch_size
    self.thread_num = thread_num
    self.dir_A = os.path.join(dataroot,'A')
    self.dir_B=os.path.join(dataroot,'B')
    self.record_list_A=os.listdir(self.dir_A)
    self.record_list_B=os.listdir(self.dir_B)
    self.A_size,self.B_size=len(self.record_list_A),len(self.record_list_B)
    self.record_number=max( self.A_size,self.B_size)
    self.record_queue = Queue(maxsize=500)
    self.image_queue=Queue(maxsize=64)

    self.record_point = 0
    t_record_producer = Thread(target=self.record_producer)
    t_record_producer.daemon = True
    t_record_producer.start()

    for i in range(self.thread_num):
      t = Thread(target=self.record_customer)
      t.daemon = True
      t.start()

  def record_producer(self):
    """record_queue's processor
    """
    while True:
      if self.record_point % self.record_number == 0:
        random.shuffle(self.record_list_A)
        random.shuffle(self.record_list_B)
        self.record_point = 0
      self.record_queue.put([self.record_list_A[self.record_point%self.A_size],self.record_list_B[self.record_point%self.B_size]])
      self.record_point += 1

  def record_process(self, item):
    imgA=imread(os.path.join(self.dir_A,item[0]))
    imgB=imread(os.path.join(self.dir_B,item[1]))
    imgA = imresize(imgA, (512,512))/255.0
    imgB = imresize(imgB, (512, 512))/255.0
    return [imgA,imgB]

  def record_customer(self):
    """record queue's customer
    """
    while True:
      item = self.record_queue.get()
      out = self.record_process(item)
      self.image_queue.put(out)

  def batch(self):
    while True:
      imagesA,imagesB=[],[]
      for i in range(self.batch_size):
        imageA,imageB=self.image_queue.get()
        imagesA.append(imageA)
        imagesB.append(imageB)
      imagesA=np.asarray(imagesA,dtype=np.float32)
      imagesB=np.asarray(imagesB,dtype=np.float32)
      return (imagesA,imagesB)


