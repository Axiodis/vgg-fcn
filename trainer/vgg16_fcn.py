import numpy as np
import tensorflow as tf
from math import ceil
from StringIO import StringIO
from tensorflow.python.lib.io import file_io

VGG_MEAN = [103.939, 116.779, 123.68]# Mean value of pixels in R G and B channels


class VGG16_FCN:
    def __init__(self, x, num_classes, keep_prob, vgg16_npy_path=None):
        
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        
        self.SKIP_LAYERS = ['fc8']
        
        self.build()
        
    def build(self):
        
        x = self.X
        #------------Build VGG16-FCN normal layers--------------------------
        
        #Layer 1
        self.conv1_1 = self.conv(x, 3, 3, 64, 1, 1, "conv1_1")
        self.conv1_2 = self.conv(self.conv1_1, 3, 3, 64, 1, 1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 2, 2, 2, 2, 'pool1')
        
        #Layer 2
        self.conv2_1 = self.conv(self.pool1, 3, 3, 128, 1, 1, "conv2_1")
        self.conv2_2 = self.conv(self.conv2_1, 3, 3, 128, 1, 1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 2, 2, 2, 2, 'pool2')
        
        #Layer 3
        self.conv3_1 = self.conv(self.pool2, 3, 3, 256, 1, 1, "conv3_1")
        self.conv3_2 = self.conv(self.conv3_1, 3, 3, 256, 1, 1, "conv3_2")
        self.conv3_3 = self.conv(self.conv3_2, 3, 3, 256, 1, 1, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 2, 2, 2, 2, 'pool3')
        
        #Layer 4
        self.conv4_1 = self.conv(self.pool3, 3, 3, 512, 1, 1, "conv4_1")
        self.conv4_2 = self.conv(self.conv4_1, 3, 3, 512, 1, 1, "conv4_2")
        self.conv4_3 = self.conv(self.conv4_2, 3, 3, 512, 1, 1, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 2, 2, 2, 2, 'pool4')
        
        #Layer 5
        self.conv5_1 = self.conv(self.pool4, 3, 3, 512, 1, 1, "conv5_1")
        self.conv5_2 = self.conv(self.conv5_1, 3, 3, 512, 1, 1, "conv5_2")
        self.conv5_3 = self.conv(self.conv5_2, 3, 3, 512, 1, 1, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 2, 2, 2, 2, 'pool5')
    
        #-----------------------Build VGG16-FCN fully connvolutional layers---------
        
        self.conv6 = self.conv(self.pool5, 7, 7, 4096, 1, 1, "fc6")
        self.drop6 = self.dropout(self.conv6, self.KEEP_PROB)
        
        self.conv7 = self.conv(self.drop6, 1, 1, 4096, 1, 1, "fc7")
        self.drop7 = self.dropout(self.conv7, self.KEEP_PROB)
        
        self.score_fr = self.conv(self.drop7, 1, 1, self.NUM_CLASSES, 1, 1, "fc8", 
                                  w_stddev = (2 / self.drop7.get_shape()[3].value)**0.5, 
                                  b_constant = 0.0, relu = False)
        
        self.pred = tf.argmax(self.score_fr, dimension=3)
        
        
        #-----------------------Build VGG16-FCN upsample layers---------
        
        self.upscore2 = self.deconv(self.score_fr,
                                            shape=tf.shape(self.pool4),
                                            num_classes = self.NUM_CLASSES,
                                            name='upscore2',
                                            ksize=4, stride=2)
        self.score_pool4 = self.conv(self.pool4, 1, 1, self.NUM_CLASSES, 1, 1, "score_pool4", 
                                     w_stddev = 0.001, 
                                     b_constant = 0.0, relu = False)
        self.fuse_pool4 = tf.add(self.upscore2, self.score_pool4)

        self.upscore4 = self.deconv(self.fuse_pool4,
                                            shape=tf.shape(self.pool3),
                                            num_classes = self.NUM_CLASSES,
                                            name='upscore4',
                                            ksize=4, stride=2)
        self.score_pool3 = self.conv(self.pool3, 1, 1, self.NUM_CLASSES, 1, 1, "score_pool3", 
                                     w_stddev = 0.0001, 
                                     b_constant = 0.0, relu = False)
        self.fuse_pool3 = tf.add(self.upscore4, self.score_pool3)

        self.upscore8 = self.deconv(self.fuse_pool3,
                                             shape=tf.shape(x),
                                             num_classes = self.NUM_CLASSES,
                                             name='upscore8',
                                             ksize=16, stride=8)

        self.pred_up = tf.argmax(self.upscore8, dimension=3)
        
        
        
    def deconv(self, x, shape, num_classes, name, ksize=4, stride=2):
        
        strides = [1, stride, stride, 1]
        
        with tf.variable_scope(name):
            in_features = x.get_shape()[3].value

            if shape is None:
                in_shape = tf.shape(x)

                h = ((in_shape[1] - 1) * stride) + 1
                w = ((in_shape[2] - 1) * stride) + 1
                new_shape = [in_shape[0], h, w, num_classes]
            else:
                new_shape = [shape[0], shape[1], shape[2], num_classes]
                
            output_shape = tf.stack(new_shape)

            f_shape = [ksize, ksize, num_classes, in_features]
            weights = self.get_deconv_filter(f_shape)
            
            deconv = tf.nn.conv2d_transpose(x, weights, output_shape, strides=strides, padding='SAME')

        return deconv

    def get_deconv_filter(self, f_shape):
        width = f_shape[0]
        height = f_shape[1]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights, dtype=tf.float32)
        var = tf.get_variable(name="up_filter", initializer=init, shape=weights.shape)
        return var
        
    def conv(self, x, filter_height, filter_width, num_filters, stride_y, stride_x, name, 
             w_stddev = None, b_constant = None, relu = True, padding='SAME'):
        
        input_channels = int(x.get_shape()[-1])
        w_initializer = None
        b_initializer = None
        
        if w_stddev is not None:
            w_initializer = tf.truncated_normal_initializer(stddev=w_stddev)
    
        if b_constant is not None:
            b_initializer = tf.constant_initializer(b_constant)
    
        with tf.variable_scope(name, reuse = tf.AUTO_REUSE) as scope:
            
            
            
            weights = tf.get_variable('weights', shape=[filter_height,
                                                        filter_width,
                                                        input_channels,
                                                        num_filters],
                                                initializer=w_initializer)
    
            biases = tf.get_variable('biases', shape=[num_filters], initializer = b_initializer)
    
        
        conv = tf.nn.conv2d(x, weights, strides=[1, stride_y, stride_x, 1], padding=padding)
    
        bias = tf.nn.bias_add(conv, biases)
        
        if relu:
            relu = tf.nn.relu(bias, name=scope.name)

            return relu
        
        return bias
    
    def max_pool(self, x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
        
        return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1],
                              padding=padding, name=name)
        
    def dropout(self, x, keep_prob):
        return tf.nn.dropout(x, keep_prob)

    def load_initial_weights(self, session, vgg16_npy_path):
        
        if(vgg16_npy_path.startswith("gs://")):
            file = file_io.read_file_to_string(vgg16_npy_path)
            weights_dict = np.load(StringIO(file), encoding='latin1').item()
        else:
            weights_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        
        for op_name in weights_dict:
            
            if op_name not in self.SKIP_LAYERS:
    
                with tf.variable_scope(op_name, reuse = tf.AUTO_REUSE):
                    
                    """Need to reshape fc layers weights"""
                    if(op_name == 'fc6' or op_name == 'fc7'):
                        var = tf.get_variable('weights')
                        shape = var.get_shape();
                        
                        weights = weights_dict[op_name][0]
                        weights = weights.reshape(shape)
                        
                        init = tf.constant_initializer(value=weights, dtype=tf.float32)
                        weights = tf.get_variable(name="weights", initializer=init, shape=shape)
                        session.run(var.assign(weights))
                        
                        var2 = tf.get_variable('biases')
                        session.run(var2.assign(weights_dict[op_name][1]))
                    else:
                        for data in weights_dict[op_name]:
        
                            if len(data.shape) == 1:
                                var = tf.get_variable('biases')
                                session.run(var.assign(data))
        
                            else:
                                var = tf.get_variable('weights')
                                session.run(var.assign(data))