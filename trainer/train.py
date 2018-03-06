from trainer.vgg16_fcn import VGG16_FCN
from trainer.datagenerator import ImageDataGenerator

import tensorflow as tf
from tensorflow.contrib.data import Iterator
from datetime import datetime
import os
import logging
import subprocess
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
import random

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('main_dir', 'main', 'Main Directory.')

log_file = "{}.log".format(datetime.now().strftime("%d-%m-%Y"))
logging.basicConfig(filename = log_file, format='%(levelname)s (%(asctime)s): %(message)s', level = logging.INFO)

num_epochs = 1000
NUM_CLASSES = 20
learning_rate = 1e-6
batch_size = 1

filewriter_path = os.path.join(FLAGS.main_dir,"vgg_fcn/tensorboard")
checkpoint_path = os.path.join(FLAGS.main_dir,"vgg_fcn/checkpoints")

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

train_file = 'train.txt'

subprocess.check_call(['gsutil', '-m' , 'cp', '-r', os.path.join(FLAGS.main_dir, "vgg_fcn", train_file), '/tmp'])

train_file = os.path.join('/tmp',train_file)

#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.955

""" Initialize datagenerator """
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 shuffle=True)

    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()

training_init_op = iterator.make_initializer(tr_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, shape=[batch_size, None, None, 3], name="input_image")
y = tf.placeholder(tf.int32, shape=[batch_size, None, None], name="input_label")
keep_prob = tf.placeholder(tf.float32)

"""Build Model"""
print("[MODEL] => Time: {} Building".format(datetime.now()))
model = VGG16_FCN(x, NUM_CLASSES, keep_prob)
logging.info("Model build")


"""Define loss function"""
loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = model.upscore8, name="Loss")))


"""Define training op"""
trainable_var = tf.trainable_variables() # Collect all trainable variables for the net

optimizer = tf.train.AdamOptimizer(learning_rate)

grads = optimizer.compute_gradients(loss, var_list=trainable_var)

train_op = optimizer.apply_gradients(grads)

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize saver for store model checkpoints
saver = tf.train.Saver()


"""Start Tensorflow session"""
print("[TRAIN] => Time: {} Start session".format(datetime.now()))
logging.info("Session started")

#try:
with tf.Session() as sess:
 
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
  
    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)
    
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    else:
        # Load the pretrained weights into the non-trainable layer
        model.load_initial_weights(sess,os.path.join(FLAGS.main_dir,"vgg_fcn/vgg16.npy"))
        print("Initial weights loaded...")
  
    print("[TRAIN] => Time: {} Start training...".format(datetime.now()))
    print("[TENSORBOARD] => Open Tensorboard at --logdir {}".format(filewriter_path))
    logging.info("Training started")

    for epoch in range(num_epochs):
        
        # Initialize iterator with the training dataset
        sess.run(training_init_op)
    
        print("[EPOCH] => Time: {} Epoch number: {}".format(datetime.now(), epoch+1))
        logging.info("Epoch: {}".format(epoch+1))
        
        with open(train_file, 'r') as f:
            lines = f.readlines()
            
        for step in range(tr_data.data_size):
            
            batch_xs, batch_ys = sess.run(next_batch)
            
            sess.run(train_op, feed_dict={x: batch_xs, 
                                          y: batch_ys, 
                                          keep_prob: 0.5})
        
        if(epoch % 20 == 0 and epoch > 0):
        
            print("[SAVE] => Time: {} Saving checkpoint of model...".format(datetime.now()))  
            
            checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(datetime.now())+'.ckpt')
            save_path = saver.save(sess, checkpoint_name)  
            
            print("[SAVE] => Time: {} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
            
    print("[TRAIN] => Time: {} Finish training...".format(datetime.now()))
    logging.info("Training finished")

    checkpoint_name = os.path.join(checkpoint_path, 'final_model'+str(datetime.now())+'.ckpt')
    print("[FINAL-SAVE] => Time: {} Final model checkpoint saved at {}".format(datetime.now(), checkpoint_name))

"""
except Exception as e:
    print("[ERROR] => Time: {} Unexpected error encountered. Please check the log file.".format(datetime.now()))
    logging.error("Error message: {}".format(e))
    logging.info("Terminating...".format(e))
"""
    
    
    
    
    