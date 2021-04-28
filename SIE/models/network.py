import numpy as np
import tensorflow as tf
import re

# ----------------------------------------------------------------------------------
# Commonly used layers and operations based on ethereon's implementation 
# https://github.com/ethereon/caffe-tensorflow
# Slight modifications may apply. FCRN-specific operations have also been appended by Iro Laina. 
# ----------------------------------------------------------------------------------
# Thanks to *Helisa Dhamo* for the model conversion and integration into TensorFlow.
# ----------------------------------------------------------------------------------

DEFAULT_PADDING = 'SAME'
DEFAULT_TYPE = tf.float32
summary = True
def ActivationSummary(layer): #tensorBoard (jmfacil)
    if summary:
        TOWER_NAME = 'tower'
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', layer.op.name)
        tf.summary.histogram(tensor_name + '/activations', layer)

'''def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")


def interleave(tensors, axis):
    old_shape = get_incoming_shape(tensors[0])[1:]
    new_shape = [-1] + old_shape
    new_shape[axis] *= len(tensors)
    return tf.reshape(tf.stack(tensors, axis + 1), new_shape)'''

def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))

        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):

    def __init__(self, inputs, batch, trainable=False,reuse=None, wd=0.0005, loss_name = "LOSS", 
            pretrained = False,
            finetuning = False,
            is_training = True):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Batch size needs to be set for the implementation of the interleaving
        self.batch_size = batch
        self.reuse = reuse 
        self.wd = wd # weight decay = prevent the weights from growing too large
        self.loss_name = loss_name
        self.theta = []
        self.theta_dict = {} 

        self.pretrained = pretrained
        self.finetuning = finetuning
        self.is_training = is_training
        
        self.setup()


    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        print(data_path)
        data_dict = np.load(data_path, encoding='latin1').item()
        for op_name in data_dict:

            with tf.variable_scope(self.nname+"/"+op_name, reuse=True):   #(op_name, reuse=True):
                for param_name, data in iter(data_dict[op_name].items()):
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        #print "Setting "+ self.nname+"/"+op_name+"/"+param_name

                    except ValueError:
                        if not ignore_missing:
                            raise
        print ("Loaded weitghts")

    def save(self, data_path, session): #chema
        to_save = {}
        for k1 in self.theta_dict.keys():
            to_save[k1] = {}
            for k2 in self.theta_dict[k1].keys():
                to_save[k1][k2] = session.run(self.theta_dict[k1][k2])
        np.save(data_path,to_save)
        return to_save

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_layer_output(self, name):
        return self.layers[name]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)
        
    def make_var(self, name, shape, init = None,wd=False,p_trainable = True):
        '''Creates a new TensorFlow variable.'''
        # Initializer param added by jmfacil
        if init is None:
            init = self.filler({"value":0.0,"type":"constant"})
        var = tf.get_variable(name, shape, dtype = 'float32', trainable=(self.trainable and p_trainable), initializer = init)
        if wd and p_trainable:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd)
            tf.add_to_collection(self.loss_name, weight_decay)
        return var

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')

    def filler(self, params): #chema
        #print "Filler: "+str(params)
        value = params.get("value",0.0)
        mean = params.get("mean",0.0)
        std = params.get("std",0.1)
        dtype = params.get("dtype",DEFAULT_TYPE)
        name = params.get("name",None)
        uniform = params.get("uniform",False)
        return {
                "xavier_conv2d" : tf.contrib.layers.xavier_initializer_conv2d(uniform = uniform),
                "t_normal" : tf.truncated_normal_initializer(mean = mean, stddev = std, dtype = dtype) ,
                "constant" : tf.constant_initializer(value = value, dtype = dtype)
                }[params.get("type","t_normal")]

    @layer
    def conv(self,
             input_data,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True,
             std=0.1,
             trainable = True,
             dropconnect = None):

        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input_data.get_shape()[-1]

        if (padding == 'SAME'):
            input_data = tf.pad(input_data, [[0, 0], [(k_h - 1)//2, (k_h - 1)//2], [(k_w - 1)//2, (k_w - 1)//2], [0, 0]], "CONSTANT")
        
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding='VALID')
        
        with tf.variable_scope(name,reuse = self.reuse) as scope: # jmfacil add reuse=True for siamese case
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i // group, c_o],init=self.filler({ "type" : "t_normal",
	                                                                                        "mean" : 0.0,
	                                                                                        "std"  : std
	                                                                                        }),wd=True)

            if trainable:
                self.theta.append(kernel)
            self.theta_dict[name] = {}
            self.theta_dict[name]["weights"] = kernel

            if dropconnect is not None:
                _op = 2
                if _op==1:
                    print ("including")
                    kernel = tf.nn.dropout(kernel, dropconnect, name="weights-dropconnect")#jmfacil
                elif _op == 2:
                    mask = tf.nn.dropout(tf.ones([1,1,c_i // group,1],tf.float32), dropconnect, name="dropconnect")#jmfacil
                    kernel = tf.multiply(kernel,mask)

            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input_data, kernel)
            else:
                # Split the input into groups and then convolve each of them independently

                input_groups = tf.split(3, group, input_data)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)

            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                if trainable:
                    self.theta.append(biases)
                self.theta_dict[name]["biases"] = biases
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            ActivationSummary(output)
            return output
            
    @layer
    def conv_trans(self,
             input_data,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True,
             std=0.01,
             trainable = True,
             dropconnect = None, dropkernel = None):

        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input_data.get_shape()[-1]
        h_i = input_data.get_shape()[1]
        w_i = input_data.get_shape()[2]

        #if (padding == 'SAME'):
        #    input_data = tf.pad(input_data, [[0, 0], [(k_h - 1)//2, (k_h - 1)//2], [(k_w - 1)//2, (k_w - 1)//2], [0, 0]], "CONSTANT")
        
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0

        def deconv_length(dim_size, stride_size, kernel_size, padding):
            if dim_size is None:
                return None
            if padding == 'VALID':
                dim_size = dim_size * stride_size + max(kernel_size - stride_size, 0)
            elif padding == 'FULL':
                dim_size = dim_size * stride_size - (stride_size + kernel_size - 2)
            elif padding == 'SAME':
                dim_size = dim_size * stride_size
            return dim_size
        # Infer the dynamic output shape:
        out_height = deconv_length(h_i,
                                  s_h, k_h,
                                  padding)
        out_width = deconv_length(w_i,
                                  s_w, k_w,
                                  padding)
        output_shape = tf.stack([self.batch_size,out_height,out_width,c_o])
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d_transpose(i, k, output_shape, [1, s_h, s_w, 1], padding=padding)
        
        with tf.variable_scope(name,reuse = self.reuse) as scope: # jmfacil add reuse=True for siamese case
            kernel = self.make_var('weights', shape=[k_h, k_w,c_o, c_i // group],init=self.filler({ "type" : "t_normal",
	                                                                                        "mean" : 0.0,
	                                                                                        "std"  : std
	                                                                                        }),wd=True)

            if trainable:
                self.theta.append(kernel)
            self.theta_dict[name] = {}
            self.theta_dict[name]["weights"] = kernel


            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                print (kernel.get_shape())
                print (c_o)
                print (c_i)
                output = convolve(input_data, kernel)
            else:
                # Split the input into groups and then convolve each of them independently

                input_groups = tf.split(3, group, input_data)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)

            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                if trainable:
                    self.theta.append(biases)
                self.theta_dict[name]["biases"] = biases
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            ActivationSummary(output)
            return output
            
    @layer
    def mul_grad(self,input_data, mul,name):
        return (1.0-mul)*tf.stop_gradient(input_data)+(mul)*input_data                
     

    @layer
    def tanh(self, input_data, name):
        return tf.nn.tanh(input_data, name=name)
    @layer
    def sigmoid(self, input_data, name):
        out = tf.nn.sigmoid(input_data, name=name)
        ActivationSummary(out)
        return out
    

    @layer
    def relu(self, input_data, name):
        return tf.nn.relu(input_data, name=name)
    @layer
    def maximum(self, input_data, max_val, name):
        return tf.maximum(max_val,input_data, name=name)

    @layer
    def max_pool(self, input_data, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input_data,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input_data, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input_data,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input_data, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input_data,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input_data, num_out, name, relu=True, std=None):
        with tf.variable_scope(name, reuse=self.reuse) as scope:
            input_shape = input_data.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input_data, [-1, dim])
            else:
                feed_in, dim = (input_data, input_shape[-1].value)
            if std is None:
                std = 2./(dim)
            weights = self.make_var('weights', shape=[dim, num_out], init=self.filler({ "type" : "t_normal",
	                                                                           "mean" : 0.0,
	                                                                           "std"  : std
	                                                                        }),wd=True)
            biases = self.make_var('biases', [num_out])
            self.theta_dict[name] = {}
            self.theta.append(weights)
            self.theta_dict[name]["weights"] = weights
            self.theta.append(biases)
            self.theta_dict[name]["biases"] = biases
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            ActivationSummary(fc)
            return fc

    @layer
    def softmax(self, input_data, name):
        input_shape = map(lambda v: v.value, input_data.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input_data = tf.squeeze(input_data, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(input_data, name)

    @layer
    def batch_normalization(self, input_data, name, scale_offset=True, relu=False, decay = 0.999,dropout=None):
        shape = [input_data.get_shape()[-1]]
        self.theta_dict[name] = {}
        with tf.variable_scope(name,reuse = self.reuse) as scope:
            if scale_offset:
                scale = self.make_var('scale', shape=shape,
                            init=self.filler(
                                {"type":"constant",
                                 "value": 1.0}
                            )
                        )
                offset = self.make_var('offset', shape=shape)
                #jmfacil
                self.theta.append(offset)
                self.theta_dict[name]["offset"] = offset
                self.theta.append(scale)
                self.theta_dict[name]["scale"] = scale
            else:
                scale, offset = (None, None)
        
            pop_mean = self.make_var('mean', shape=shape, p_trainable = False)
            pop_var = self.make_var('variance', shape=shape,
                            init=self.filler(
                                {"type":"constant",
                                 "value": 1.0}
                            ),
                            wd = False, p_trainable = False
                        )
            #jmfacil
            self.theta_dict[name]["variance"] = pop_var
            self.theta_dict[name]["mean"] = pop_mean

        if self.is_training:
            batch_mean, batch_var = tf.nn.moments(input_data, [0,1,2], name='moments')
            train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                epsilon=1e-4
                output =  tf.nn.batch_normalization(input_data,
                    batch_mean, batch_var, offset, scale, epsilon)
        else:
            epsilon=1e-4
            output = tf.nn.batch_normalization(input_data,
                pop_mean, pop_var, offset, scale, epsilon)

        #jmfacil: dropout added based on pix2pix
        if dropout is not None:
            output = tf.nn.dropout(output,dropout)
        if relu:
            output = tf.nn.relu(output)
        ActivationSummary(output)
        return output

    @layer
    def batch_normalization3(self, input_data, name, scale_offset=True, relu=False, decay = 0.999):
        shape = [input_data.get_shape()[-1]]
        self.theta_dict[name] = {}
        with tf.variable_scope(name,reuse = self.reuse) as scope:
            if scale_offset:
                scale = self.make_var('scale', shape=shape,
                            init=self.filler(
                                {"type":"constant",
                                 "value": 1.0}
                            )
                        )
                offset = self.make_var('offset', shape=shape)

                self.theta.append(offset)
                self.theta_dict[name]["offset"] = offset
                self.theta.append(scale)
                self.theta_dict[name]["scale"] = scale
            else:
                scale, offset = (None, None)
        
            pop_mean = self.make_var('mean', shape=shape)
            pop_var = self.make_var('variance', shape=shape,
                            init=self.filler(
                                {"type":"constant",
                                 "value": 1.0}
                            )
                        )
            self.theta_dict[name]["variance"] = pop_var
            self.theta_dict[name]["mean"] = pop_mean

        if self.is_training:
            batch_mean, batch_var = tf.nn.moments(input_data, [0,1,2], name='moments')
            train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                epsilon=1e-4
                output =  tf.nn.batch_normalization(input_data,
                    batch_mean, batch_var, offset, scale, epsilon)
        else:
            epsilon=1e-4
            output = tf.nn.batch_normalization(input_data,
                pop_mean, pop_var, offset, scale, epsilon)

        if relu:
            output = tf.nn.relu(output)
        ActivationSummary(output)
        return output

    @layer
    def batch_normalization2(self, input_data, name, scale_offset=True, relu=False):
        # NOTE: Currently, only inference is supported * and training now
        self.theta_dict[name] = {}
        with tf.variable_scope(name,reuse = self.reuse) as scope:
            shape = [input_data.get_shape()[-1]]
            if scale_offset:
                scale = self.make_var('scale', shape=shape,
                            init=self.filler(
                                {"type":"constant",
                                 "value": 1.0}
                            )
                        )
                offset = self.make_var('offset', shape=shape)

                self.theta.append(offset)
                self.theta_dict[name]["offset"] = offset
                self.theta.append(scale)
                self.theta_dict[name]["scale"] = offset
            else:
                scale, offset = (None, None)
        
        with tf.variable_scope(name,reuse = None) as scope:
            batch_mean, batch_var = tf.nn.moments(input_data, [0,1,2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean,name="mean"), tf.identity(batch_var,name="variance")
            
            mean, var = tf.cond(tf.constant(self.is_training),#self.trainable, TODO
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
            #"""self.theta.append(var)
            #self.theta_dict[name]["variance"] = var
            #self.theta.append(mean)
            #self.theta_dict[name]["mean"] = mean
            
            output = tf.nn.batch_normalization(
                input_data,
                mean=mean,#self.make_var('mean', shape=shape),
                variance=var,#self.make_var('variance', shape=shape),
                offset=offset,
                scale=scale,
				variance_epsilon=1e-4,
                name=name)
            if relu:
                output = tf.nn.relu(output)


            ActivationSummary(output)
            return output

    @layer
    def dropout(self, input_data, keep_prob, name):
        return tf.nn.dropout(input_data, keep_prob, name=name)
    

    # -------------------------------------------------------
    # Additional operations, specific to FCRN by Iro Laina
    # -------------------------------------------------------
    
    def prepare_indices(self, before, row, col, after, dims ):

        x0, x1, x2, x3 = np.meshgrid(before, row, col, after)

        x_0 = tf.Variable(x0.reshape([-1]), name = 'x_0')
        x_1 = tf.Variable(x1.reshape([-1]), name = 'x_1')
        x_2 = tf.Variable(x2.reshape([-1]), name = 'x_2')
        x_3 = tf.Variable(x3.reshape([-1]), name = 'x_3')

        linear_indices = x_3 + dims[3].value * x_2  + 2 * dims[2].value * dims[3].value * x_0 * 2 * dims[1].value + 2 * dims[2].value * dims[3].value * x_1
        linear_indices_int = tf.to_int32(linear_indices)

        return linear_indices_int

    def unpool_as_conv(self, size, input_data, id, stride = 1, ReLU = False, BN = True):

		# Model upconvolutions (unpooling + convolution) as interleaving feature
		# maps of four convolutions (A,B,C,D). Building block for up-projections. 


        # Convolution A (3x3)
        # --------------------------------------------------
        layerName = "layer%s_ConvA" % (id)
        self.feed(input_data)
        self.conv( 3, 3, size[3], stride, stride, name = layerName, padding = 'SAME', relu = False)
        outputA = self.get_output()

        # Convolution B (2x3)
        # --------------------------------------------------
        layerName = "layer%s_ConvB" % (id)
        padded_input_B = tf.pad(input_data, [[0, 0], [1, 0], [1, 1], [0, 0]], "CONSTANT")
        self.feed(padded_input_B)
        self.conv(2, 3, size[3], stride, stride, name = layerName, padding = 'VALID', relu = False)
        outputB = self.get_output()

        # Convolution C (3x2)
        # --------------------------------------------------
        layerName = "layer%s_ConvC" % (id)
        padded_input_C = tf.pad(input_data, [[0, 0], [1, 1], [1, 0], [0, 0]], "CONSTANT")
        self.feed(padded_input_C)
        self.conv(3, 2, size[3], stride, stride, name = layerName, padding = 'VALID', relu = False)
        outputC = self.get_output()

        # Convolution D (2x2)
        # --------------------------------------------------
        layerName = "layer%s_ConvD" % (id)
        padded_input_D = tf.pad(input_data, [[0, 0], [1, 0], [1, 0], [0, 0]], "CONSTANT")
        self.feed(padded_input_D)
        self.conv(2, 2, size[3], stride, stride, name = layerName, padding = 'VALID', relu = False)
        outputD = self.get_output()

        # Interleaving elements of the four feature maps
        # --------------------------------------------------
        dims = outputA.get_shape()
        dim1 = dims[1] * 2
        dim2 = dims[2] * 2

        A_row_indices = range(0, dim1, 2)
        A_col_indices = range(0, dim2, 2)
        B_row_indices = range(1, dim1, 2)
        B_col_indices = range(0, dim2, 2)
        C_row_indices = range(0, dim1, 2)
        C_col_indices = range(1, dim2, 2)
        D_row_indices = range(1, dim1, 2)
        D_col_indices = range(1, dim2, 2)

        all_indices_before = range(int(self.batch_size))
        all_indices_after = range(dims[3])

        A_linear_indices = self.prepare_indices(all_indices_before, A_row_indices, A_col_indices, all_indices_after, dims)
        B_linear_indices = self.prepare_indices(all_indices_before, B_row_indices, B_col_indices, all_indices_after, dims) 
        C_linear_indices = self.prepare_indices(all_indices_before, C_row_indices, C_col_indices, all_indices_after, dims)
        D_linear_indices = self.prepare_indices(all_indices_before, D_row_indices, D_col_indices, all_indices_after, dims)

        A_flat = tf.reshape(tf.transpose(outputA, [1, 0, 2, 3]), [-1])
        B_flat = tf.reshape(tf.transpose(outputB, [1, 0, 2, 3]), [-1])
        C_flat = tf.reshape(tf.transpose(outputC, [1, 0, 2, 3]), [-1])
        D_flat = tf.reshape(tf.transpose(outputD, [1, 0, 2, 3]), [-1])

        Y_flat = tf.dynamic_stitch([A_linear_indices, B_linear_indices, C_linear_indices, D_linear_indices], [A_flat, B_flat, C_flat, D_flat])
        Y = tf.reshape(Y_flat, shape = tf.to_int32([-1, dim1.value, dim2.value, dims[3].value]))
        
        if BN:
            layerName = "layer%s_BN" % (id)
            self.feed(Y)
            self.batch_normalization(name = layerName, scale_offset = True, relu = False)
            Y = self.get_output()

        if ReLU:
            Y = tf.nn.relu(Y, name = layerName)
        
        return Y

    @layer
    def up_project(self,
                   input_data,
                   size, name, stride = 1, BN = True):
        
        # Create residual upsampling layer (UpProjection)
        id = name
        #input_data = self.get_output()

        # Branch 1
        id_br1 = "%s_br1" % (id)

        # Interleaving Convs of 1st branch
        out = self.unpool_as_conv(size, input_data, id_br1, stride, ReLU=True, BN=True)

        # Convolution following the upProjection on the 1st branch
        layerName = "layer%s_Conv" % (id)
        self.feed(out)
        self.conv(size[0], size[1], size[3], stride, stride, name = layerName, relu = False)

        if BN:
            layerName = "layer%s_BN" % (id)
            self.batch_normalization(name = layerName, scale_offset=True, relu = False)

        # Output of 1st branch
        branch1_output = self.get_output()

            
        # Branch 2
        id_br2 = "%s_br2" % (id)
        # Interleaving convolutions and output of 2nd branch
        branch2_output = self.unpool_as_conv(size, input_data, id_br2, stride, ReLU=False)

        
        # sum branches
        layerName = "layer%s_Sum" % (id)
        output = tf.add_n([branch1_output, branch2_output], name = layerName)
        # ReLU
        layerName = "layer%s_ReLU" % (id)
        output = tf.nn.relu(output, name=layerName)

        ActivationSummary(output)
        return output

    def up_project2(self, size, id, stride = 1, BN = True):
        
        # Create residual upsampling layer (UpProjection)

        input_data = self.get_output()

        # Branch 1
        id_br1 = "%s_br1" % (id)

        # Interleaving Convs of 1st branch
        out = self.unpool_as_conv(size, input_data, id_br1, stride, ReLU=True, BN=True)

        # Convolution following the upProjection on the 1st branch
        layerName = "layer%s_Conv" % (id)
        self.feed(out)
        self.conv(size[0], size[1], size[3], stride, stride, name = layerName, relu = False)

        if BN:
            layerName = "layer%s_BN" % (id)
            self.batch_normalization(name = layerName, scale_offset=True, relu = False)

        # Output of 1st branch
        branch1_output = self.get_output()

            
        # Branch 2
        id_br2 = "%s_br2" % (id)
        # Interleaving convolutions and output of 2nd branch
        branch2_output = self.unpool_as_conv(size, input_data, id_br2, stride, ReLU=False)

        
        # sum branches
        layerName = "layer%s_Sum" % (id)
        output = tf.add_n([branch1_output, branch2_output], name = layerName)
        # ReLU
        layerName = "layer%s_ReLU" % (id)
        output = tf.nn.relu(output, name=layerName)

        ActivationSummary(output)
        self.feed(output)
        return self

