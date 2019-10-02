import logging

import tensorflow as tf

from learn.modules import cnn_module, cudnn_lstm_module, fused_lstm_module, batch_norm, resize, recompute


# Model file - define neural network architecture

def default(config, keys, default_value):
    """
    Helper function to read multi-dimensional dictionary
    :param config: input dictionary
    :param keys: keys in chronological order used to traverse multi-dimensional dictionary
    :param default_value: default value to return when keys do not exist
    :return: values of dictionary or default
    """
    c = config
    if config is None:
        print("config unset")
        return default_value
    for key in keys:
        if key in config:
            c = config[key]
        else:
            print("%s not set" % str(key))
            return default_value
    return c


class NNModel:
    """
    Neural Network model class which holds network architecture and model relevant parameters such as learning rate,
    batch size, loss definition, training op, early stopping threshold etc.
    """

    def __init__(self, config, lstm_type='cudnn'):
        self.learning_rate_initial = default(config, ['lr_start'], 0.001)
        self.learning_rate_schedule_width = default(config, ['lr_width'], 5)
        self.lstm_type = lstm_type  # defines which lstm impl. to used
        self.config = config  # configuration dictionary

    def arch(self, input, train):
        """
        Actual neural network architecture - built from configuration dictionary read from config file (e.g. deepcci.json)
        :param input: input tensor (typically batched dataset entries) - shape: [batch_size x length x features]
        :param train: is_train placeholder
        :return: output logits of neural network
        """
        a = input

        # find out which lstm implementation to use
        if self.lstm_type == 'cudnn':
            logging.info("use cudnn lstm")
            lstm_module = cudnn_lstm_module
        else:
            logging.info("use TF fused lstm")
            lstm_module = fused_lstm_module

        # map config file type names to neural network module
        mapping_function = {"bn": batch_norm, "resize": resize, "cnn_module": cnn_module, "lstm": lstm_module}

        # read architecture from model file and build it
        model = self.config['model']
        for p in model:
            type = p['type']
            func_params = {}
            func_params.update(p)
            del func_params['type']
            if 'recompute' in func_params:
                del func_params['recompute']
                a = recompute(a, mapping_function[type], func_params['name'], train, func_params)
            else:
                a = mapping_function[type](a, train=train, recomp=False, **func_params)

        a = tf.layers.batch_normalization(a, training=train, name="logits_bn")
        a = tf.layers.conv1d(inputs=a, filters=5, kernel_size=1, strides=1, padding='same',
                             name="conv1d_last", kernel_initializer=tf.contrib.layers.xavier_initializer())
        return a

    def loss(self, logits, variant, pacing, alpha):
        """
        compute loss as described in paper
        :param logits: input logits from neural network
        :param variant: variant labels to logits
        :param pacing: pacing label to logits
        :param alpha: loss alpha as described in paper
        :return: loss
        """

        # split logits into variant and pacing labels
        logvariant = self.extract_variant_logits(logits)
        logpacing = self.extract_pacing_logits(logits)

        # get logit shape for reshaping label vector
        s = tf.shape(logits)

        # flatten logits batch and length into one dimension (e.g. 30 x 60 x 3 -> 1800 x 3) examppe loss1
        logvariant = tf.reshape(logvariant, [-1, 3])
        logpacing = tf.reshape(logpacing, [-1, 2])

        # reshape labels over logits (e.g. for batch size 30, 30 different labels exist.
        # However, we get several logits after every LSTM time step (for example in loss1, we have 60 timesteps)
        # Therefore, we need to stretch labels over 60 timesteps. So we get 30 x 60 labels.
        # Then we also flatten these as the logits form final softmax crossentropy
        labels = tf.reshape(tf.tile(tf.expand_dims(variant, 1), [1, s[1]]), [-1])
        pacing = tf.reshape(tf.tile(tf.expand_dims(pacing, 1), [1, s[1]]), [-1])

        # return averaged, weighted crossentropy loss over softmax via sparse_softmax_cross_entropy_with_logits
        return (1 - alpha) * tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logvariant, labels=labels)) + alpha * tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logpacing, labels=pacing))

    # when extending the model, these following functions need to be adapted
    def num_classes(self):
        """
        total number of classes for confusion matrix
        :return:
        """
        return 6

    def extract_variant_logits(self, logits):
        """
        extracting variant logits
        :param logits:
        :return:
        """
        return logits[:, :, 0:3]

    def extract_pacing_logits(self, logits):
        """
        extracting pacing logits from final classification layer
        :param logits:
        :return:
        """
        return logits[:, :, 3:5]

    def predicted_variant(self, logits):
        """
        extract variant part of logits - we use a classification probability threshold
        :param logits: logit output of neural network
        :return: variant part of logits
        """

        prob_threshold = 0.5

        logvariant = self.extract_variant_logits(logits)

        softmax = tf.nn.softmax(logvariant)
        argmax = tf.argmax(logvariant, axis=-1, output_type=tf.int32)

        prob_threshold_exceeded = tf.reduce_max(tf.cast(softmax > prob_threshold, tf.int32), axis=-1)

        threes = tf.tile([[3]], tf.shape(argmax))  # tensor of shape as argmax only filled with 3

        return tf.where(prob_threshold_exceeded > 0, argmax, threes)

    def combine_labels(self, variantlabel, pacinglabel):
        """
        combines variant and pacing label into a combined label
        0 0 -> 0 bbr, 1 0 -> 1 cubic, 2 0 - > 2 reno, 1 1 -> 3 cubic-pacing, 2 1 -> 4 reno-pacing, 0 1 -> 5, 3 x -> 5
        :param variantlabel: tensor of variant labels
        :param pacinglabel: tensor of pacing labels
        :return: combined label
        """
        # when variant is 3, our classification is unknown - map to unknown label 5
        # when variant is 0 (bbr) and pacing is 1, it would be bbr-pacing - however, we train bbr without pacing label
        # so bbr-pacing is unknown to our network and we label it as 5 unknown

        unknown_label = tf.tile([5], tf.shape(variantlabel))  # tensor of shape as variantlabel filled with 5

        combined_label = tf.where(variantlabel > 2, unknown_label, variantlabel + pacinglabel * 2)
        return tf.where(variantlabel < 1, tf.where(pacinglabel > 0, unknown_label, combined_label), combined_label)

    def predicted_pacing(self, logits):
        """
        extract pacing part of logits
        :param logits: logit output of neural network
        :return: pacing part of logits
        """
        # in binary case, threshold is always exceeded
        return tf.argmax(self.extract_pacing_logits(logits), axis=-1, output_type=tf.int32)

    def get_trainbatchsize(self):
        return default(self.config, ['train_batch_size'], 32)

    def get_testbatchsize(self):
        return default(self.config, ['test_batch_size'], 48)

    def get_maxepoch(self):
        return default(self.config, ['max_num_epoch'], 50)

    def get_lossalpha(self):
        return default(self.config, ['loss_alpha'], 0.2)

    def get_earlystopping(self):
        return default(self.config, ['early_stopping'], 5)

    def optimize(self, loss, learning_rate):
        """
        Adam optimizer part with gradient clipping to avoid vanishing gradients (should not occur with LSTMs, but for safety)
        :param loss: loss
        :param learning_rate: learning rate
        :return: train opp
        """

        # bases in parts on https://stackoverflow.com/questions/36498127/how-to-apply-gradient-clipping-in-tensorflow
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradients = optimizer.compute_gradients(loss)
            #        var_list = (tf.trainable_variables() + tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
            #        g = tf.gradients(loss, var_list)
            #        g = list(zip(g, var_list))
            gradients = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gradients]  # clip gradients
            train_op = optimizer.apply_gradients(gradients, tf.train.get_or_create_global_step())
        # train_op = optimizer.minimize(loss, global_tr_step)
        return train_op

    def learning_rate_schedule(self, epoch):
        """
        learning rate schedule - halves learning rate after x epochs
        :param epoch: current epoch
        :return: current learning rate
        """
        e = epoch // self.learning_rate_schedule_width
        return self.learning_rate_initial * (0.5 ** e)

    def set_lstmtype(self, lstm_type):
        """
        overwrite lstm_type to use
        :param lstm_type: can be 'cudnn' or something else (e.g. 'fused')
        """
        self.lstm_type = lstm_type
