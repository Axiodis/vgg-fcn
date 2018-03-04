from trainer.vgg16_fcn import VGG16_FCN
from trainer.data_reader import DataReader
from datetime import datetime
import tensorflow as tf
import os
import numpy as np
import logging

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('main_dir', 'main', 'Main Directory.')

log_file = "{}.log".format(datetime.now().strftime("%d-%m-%Y"))
logging.basicConfig(filename = log_file, format='%(levelname)s (%(asctime)s): %(message)s', level = logging.INFO)

num_epochs = 1000
NUM_CLASSES = 20
learning_rate = 1e-6
batch_size = 2

images_dir = os.path.join(FLAGS.main_dir,"/VOC2012/JPEGImages")
labels_dir = os.path.join(FLAGS.main_dir,"/VOC2012/SegmentationClass")

filewriter_path = os.path.join(FLAGS.main_dir,"/vgg_fcn/tensorboard")
checkpoint_path = os.path.join(FLAGS.main_dir,"/vgg_fcn/checkpoints")

tf.reset_default_graph()

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

#Initialize data reader
train_generator = DataReader(images_dir,labels_dir,batch_size)
train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)



"""Start Tensorflow session"""
print("[TRAIN] => Time: {} Start session".format(datetime.now()))
logging.info("Session started")

try:
    with tf.Session() as sess:
     
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
      
        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)
      
        # Load the pretrained weights into the non-trainable layer
        model.load_initial_weights(sess,os.path.join(FLAGS.main_dir,"/vgg_fcn/vgg16.npy"))
      
        print("[TRAIN] => Train batches per epoch: {}".format(train_batches_per_epoch))
        print("[TRAIN] => Time: {} Start training...".format(datetime.now()))
        print("[TENSORBOARD] => Open Tensorboard at --logdir {}".format(filewriter_path))
        logging.info("Training started")
    
        for epoch in range(num_epochs):
        
            print("[EPOCH] => Time: {} Epoch number: {}".format(datetime.now(), epoch+1))
            logging.info("Epoch: {}".format(epoch+1))
            
            step = 1
            
            while step < train_batches_per_epoch:
                
                print("Step {} of {}".format(step, train_batches_per_epoch))
                
                with tf.device('/cpu:0'):
                    batch_xs, batch_ys = train_generator.next_batch()
                
                sess.run(train_op, feed_dict={x: batch_xs, 
                                              y: batch_ys, 
                                              keep_prob: 0.5})
                step += 1
                
            train_generator.reset_pointer()
            
            if(epoch % 25 == 0 and epoch > 0):
            
                print("[SAVE] => Time: {} Saving checkpoint of model...".format(datetime.now()))  
                
                checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(datetime.now())+'.ckpt')
                save_path = saver.save(sess, checkpoint_name)  
                
                print("[SAVE] => Time: {} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
                
        print("[TRAIN] => Time: {} Finish training...".format(datetime.now()))
        logging.info("Training finished")
    
        checkpoint_name = os.path.join(checkpoint_path, 'final_model'+str(datetime.now())+'.ckpt')
        print("[FINAL-SAVE] => Time: {} Final model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
                
except Exception as e:
    print("[ERROR] => Time: {} Unexpected error encountered. Please check the log file.".format(datetime.now()))
    logging.error("Error message: {}".format(e))
    logging.info("Terminating...".format(e))
        
    
    
    
    
    