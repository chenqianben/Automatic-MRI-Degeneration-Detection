import tensorflow as tf;
import numpy as np;


class AffineLayer(tf.keras.layers.Layer):
    '''这个class的作用是给定一个输入和affine transformation（2x3的矩阵），得到变换的输出（需要double linear interpolation'''
    def __init__(self, downsample_factor = 1):

        # output resolution downgration.
        super(AffineLayer, self).__init__();
        self.downsample_factor = downsample_factor;

    def call(self, inputs, affines):

        # affine transform output grid to the input grid and extract intensity back to output
        assert len(inputs.shape) == 4;
        assert len(affines.shape) == 3 and affines.shape[-2:] == (2,3);
        assert inputs.shape[0] == affines.shape[0];
        
        if inputs.dtype != tf.float32: inputs = tf.cast(inputs, dtype = tf.float32);
        input_shape = tf.shape(inputs);
        # meshgrid of normalized coordinates in output resolution: [x,y] = meshgrid(x,y)
        x = tf.ones(shape = (input_shape[1]//self.downsample_factor, 1), dtype = tf.float32) * tf.reshape(tf.linspace(-1.,1.,input_shape[2]//self.downsample_factor), shape = (1,input_shape[2]//self.downsample_factor));
        y = tf.ones(shape = (1, input_shape[2]//self.downsample_factor), dtype = tf.float32) * tf.reshape(tf.linspace(-1.,1.,input_shape[1]//self.downsample_factor), shape = (input_shape[1]//self.downsample_factor,1));
        # homogeneous normalized coordinates of meshgrid
        # homogeneous.shape = (3, output_element_num)
        x_flat = tf.reshape(x,shape = (-1,));
        y_flat = tf.reshape(y,shape = (-1,));
        ones = tf.ones_like(x_flat);
        homogeneous = tf.stack([x_flat, y_flat, ones]);
        # affine transform the normalized coordinates
        # trans_xy.shape = (batch_num, 2, output_element_num)
        trans_xy = tf.linalg.matmul(affines, tf.tile(tf.expand_dims(homogeneous, axis = 0), (input_shape[0], 1, 1)))
        # convert normalized coordinates to input coordinates
        # (x|y).shape = (batch_num, output_element_num)
        zero = tf.zeros(shape = (), dtype = tf.int32);
        max_y = tf.cast(input_shape[1] - 1, dtype = tf.int32);
        max_x = tf.cast(input_shape[2] - 1, dtype = tf.int32);
        x = (trans_xy[:,0,:] + 1.0) / 2. * tf.cast(input_shape[2], dtype = tf.float32);
        y = (trans_xy[:,1,:] + 1.0) / 2. * tf.cast(input_shape[1], dtype = tf.float32);
        # four nearest coordinates on the input grid
        # (xl|xr|yu|yb).shape = (batch_num, output_element_num)
        xl = tf.clip_by_value(tf.cast(tf.floor(x), dtype = tf.int32), zero, max_x);
        xr = tf.clip_by_value(xl + 1, zero, max_x);
        yu = tf.clip_by_value(tf.cast(tf.floor(y), dtype = tf.int32), zero, max_y);
        yb = tf.clip_by_value(yu + 1, zero, max_y);
        # every row of tensor base is a row of start address of every batch of input image
        # base.shape = (batch_num, output_element_num)
        base = tf.reshape(tf.range(input_shape[0], dtype = tf.int32) * input_shape[1] * input_shape[2], shape = (-1, 1)) * tf.ones(shape = (1, input_shape[1]//self.downsample_factor * input_shape[2]//self.downsample_factor), dtype = tf.int32);
        # four nearest coordinates on the input grid of every batch
        # (lu|lb|ru|rb)_index.shape = (batch_num, output_element_num)
        lu_index = base + yu * input_shape[2] + xl;
        lb_index = base + yb * input_shape[2] + xl;
        ru_index = base + yu * input_shape[2] + xr;
        rb_index = base + yb * input_shape[2] + xr;
        # extract values of the four nearest coordinates on the input grid of every batch
        # (lu|lb|ru|rb)_value.shape = (batch_num, output_element_num, channel_num)
        im_flat = tf.reshape(inputs, shape = (-1, input_shape[3])); # im_flat.shape = (batch_num * output_element_num, channel_num)
        lu_value = tf.reshape(tf.gather(im_flat,tf.reshape(lu_index, shape = (-1,))), shape = (-1, input_shape[1]//self.downsample_factor * input_shape[2]//self.downsample_factor, input_shape[3]));
        lb_value = tf.reshape(tf.gather(im_flat,tf.reshape(lb_index, shape = (-1,))), shape = (-1, input_shape[1]//self.downsample_factor * input_shape[2]//self.downsample_factor, input_shape[3]));
        ru_value = tf.reshape(tf.gather(im_flat,tf.reshape(ru_index, shape = (-1,))), shape = (-1, input_shape[1]//self.downsample_factor * input_shape[2]//self.downsample_factor, input_shape[3]));
        rb_value = tf.reshape(tf.gather(im_flat,tf.reshape(rb_index, shape = (-1,))), shape = (-1, input_shape[1]//self.downsample_factor * input_shape[2]//self.downsample_factor, input_shape[3]));
        # calculate weights of four nearest coordinates
        # (lu|lb|ru|rb)_weight.shape = (batch_num, output_element_num,1)
        lu_weight = tf.expand_dims(tf.math.exp(-(x - tf.cast(xl, dtype = tf.float32)) * (y - tf.cast(yu, dtype = tf.float32))), axis = -1);
        lb_weight = tf.expand_dims(tf.math.exp(-(x - tf.cast(xl, dtype = tf.float32)) * (tf.cast(yb, dtype = tf.float32) - y)), axis = -1);
        ru_weight = tf.expand_dims(tf.math.exp(-(tf.cast(xr, dtype = tf.float32) - x) * (y - tf.cast(yu, dtype = tf.float32))), axis = -1);
        rb_weight = tf.expand_dims(tf.math.exp(-(tf.cast(xr, dtype = tf.float32) - x) * (tf.cast(yb, dtype = tf.float32) - y)), axis = -1);
        # weighted sum values of the four nearest coordinates
        # output.shape = (batch_num, output_element_num, channel_num)
        weighted_sum = (lu_weight * lu_value + lb_weight * lb_value + ru_weight * ru_value + rb_weight * rb_value);
        weight_sum = (lu_weight + lb_weight + ru_weight + rb_weight);
        output = weighted_sum / weight_sum;
        # reshape output
        output = tf.keras.layers.Reshape((input_shape[1] // self.downsample_factor, input_shape[2] // self.downsample_factor, input_shape[3]))(output);
        return output;

    
class DefaultLocalizationNetwork(tf.keras.Model):
    '''这是一个简单的CNN的定位学习模型，输入是图片，输出是affine transformer（2x3的矩阵），目的是通过输入图片来学习变换 -> dynamic filter networks'''
    def __init__(self):

        super(DefaultLocalizationNetwork, self).__init__();
        self.conv = tf.keras.layers.Conv2D(filters = 20, kernel_size = (5,5), strides = (1,1), padding = 'valid');
        self.dense = tf.keras.layers.Dense(units = 6, kernel_initializer = tf.constant_initializer(np.zeros(shape = (20,6), dtype = np.float32)), bias_initializer = tf.constant_initializer(np.array([1, 0, 0, 0, 1, 0], dtype = np.float32)));

    def call(self, inputs):

        results = self.conv(inputs);
        results = tf.math.reduce_mean(results, axis = [1,2]);
        results = self.dense(results);
        results = tf.reshape(results, shape = (-1, 2, 3));
        return results;

    
class SpatialTransformLayer(tf.keras.layers.Layer):
    '''这是Spatial Transform Network，把之前的规整到一起'''
    def __init__(self, downsample_factor = 1, loc_model = DefaultLocalizationNetwork()):

        super(SpatialTransformLayer, self).__init__();
        self.loc_model = loc_model;
        self.affine_layer = AffineLayer(downsample_factor);

    def call(self, inputs):
        if inputs.dtype != tf.float32: inputs = tf.cast(inputs, dtype = tf.float32);
        affines = self.loc_model(inputs);   # 通过输入图片得到仿射变换
        results = self.affine_layer(inputs, affines);    # 通过输入图片和仿射变换得到输出图片
        return results;