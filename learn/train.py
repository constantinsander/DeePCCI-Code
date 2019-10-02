import argparse
import datetime
import glob
import json
import logging
import os
import socket
import sys
import uuid

import tensorflow as tf

sys.path.append("..")

import learn.dataset as datasethelper
from learn.model import NNModel
from learn.learn_utils import DatasetEntry, ResultHelper, RunHelper, cm_recall_precision_f1

from utils.utils import JsonSerialization as json

tf.logging.set_verbosity(tf.logging.ERROR)
logging.basicConfig(level=logging.DEBUG)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class DeepcciTF:
    class config:
        weight_dir = '../data/weights'
        summary_dir = '../data/log'
        dataset_expected_dimensions = 1
        dataset_cache_file = True
        dataset_cache_ram = False
        dataset_prefetchsize = 200
        dataset_shufflesize = 4
        dataset_parallel_readers = 12
        dataset_rootpath = '../data/processed'
        dataset_datapath = '1ms'
        dataset_vallabelfile = 'val.csv'
        dataset_testlabelfile = 'test.csv'
        dataset_trainlabelfile = 'train.csv'
        dataset_configpath = '../data/jsons'

    class tfvars:
        """
        saves tensorflow variables
        """
        loss = None
        dataset_handle = None
        dataset_next_entry = None
        is_train = None
        entry_begin = None
        entry_end = None
        logits = None
        learning_rate = None
        confusion_matrix = None
        pred_combined = None
        real_combined = None
        pred_pacinglabel = None
        pred_variantlabel = None
        acc_combined = None
        acc_variantlabel = None
        acc_pacinglabel = None

    def set_dataset_rootpath(self, path):
        self.config.dataset_rootpath = path

    def set_dataset_datapath(self, path):
        self.config.dataset_datapath = path

    def set_dataset_testlabelfile(self, filename):
        self.config.dataset_testlabelfile = filename

    def set_dataset_trainlabelfile(self, filename):
        self.config.dataset_trainlabelfile = filename

    def set_dataset_vallabelfile(self, filename):
        self.config.dataset_vallabelfile = filename

    def __init__(self, name, model):
        self.name = name
        self.NNModel = model

        self.uid = uuid.uuid4()
        self.extraid = None
        self.time = datetime.datetime.now()

        self.variables_initialized = False

        self.train_op = None

        self.file_loaded = None
        self.summary_writers = {}

        #setup general tensorflow placeholders
        self.tfvars.is_train = tf.placeholder(tf.bool, name='is_train') # is_train placeholder to inform batch norm about training
        self.tfvars.entry_begin = tf.placeholder(tf.int64, name='entry_begin') # begin of slice of time series entry
        self.tfvars.entry_end = tf.placeholder(tf.int64, name='entry_end') # end of slice of time series entry
        self.tfvars.learning_rate = tf.placeholder(tf.float32, shape=[]) # defines learning rate

        #tensorflow dataset operation
        self.dataset_handle = tf.placeholder(dtype=tf.string, shape=[]) # handle for tensorflow dataset

        #iterator from dataset
        iterator = tf.data.Iterator.from_string_handle(self.dataset_handle, datasethelper.get_expected_types(),
                                                       datasethelper.get_expected_output_shapes())

        #dataset entry from iterator - gives back multiple tensorflow placeholders which holds dataset entries
        #e.g. dataset_entries.sequence is placeholder which holds data of histogram [batch_size x 60000 x features] at this point
        self.dataset_entries = DatasetEntry(*iterator.get_next())

        #slice dataset entries for different time series lengths
        self.dataset_entries.sequence = self.dataset_entries.sequence[:, self.tfvars.entry_begin:self.tfvars.entry_end]
        # dataset entry at this point unaware of expected feature dimensions - set explicitly to support tensorflow
        # when setting up the neural network (e.g. convolution weights)
        self.dataset_entries.sequence.set_shape([None, None, self.config.dataset_expected_dimensions])

        # create actual neural network model (see model.py) by feeding dataset entries and training information
        self.tfvars.logits = self.NNModel.arch(self.dataset_entries.sequence, self.tfvars.is_train)
        # summary for tensorboard
        tf.summary.histogram('logits', self.tfvars.logits)

        #set up metrics
        self._get_metrics()

    def _get_metrics(self):
        """
        set up metrics (loss, accuracy, accuracy for variant, accuracy for pacing)
        """

        # loss as defined in paper
        self.tfvars.loss = self.NNModel.loss(self.tfvars.logits, self.dataset_entries.variantlabel,
                                             self.dataset_entries.pacinglabel, self.NNModel.get_lossalpha())

        # split logits into variant and pacing prediction
        self.tfvars.pred_variantlabel = self.NNModel.predicted_variant(self.tfvars.logits)
        self.tfvars.pred_pacinglabel = self.NNModel.predicted_pacing(self.tfvars.logits)

        pred_combined = self.NNModel.combine_labels(self.tfvars.pred_variantlabel[:, -1],
                                                    self.tfvars.pred_pacinglabel[:, -1])
        real_combined = self.NNModel.combine_labels(self.dataset_entries.variantlabel,
                                                    self.dataset_entries.pacinglabel)

        num_classes = self.NNModel.num_classes()

        self.tfvars.confusion_matrix = tf.confusion_matrix(real_combined, pred_combined, num_classes=num_classes)

        # compute accuracies
        corr_combined = tf.equal(real_combined, pred_combined)
        corr_variantlabel = tf.equal(self.dataset_entries.variantlabel, self.tfvars.pred_variantlabel[:, -1])
        corr_pacinglabel = tf.equal(self.dataset_entries.pacinglabel, self.tfvars.pred_pacinglabel[:, -1])

        self.tfvars.acc_combined = tf.reduce_mean(tf.cast(corr_combined, tf.float32))
        self.tfvars.acc_variantlabel = tf.reduce_mean(tf.cast(corr_variantlabel, tf.float32))
        self.tfvars.acc_pacinglabel = tf.reduce_mean(tf.cast(corr_pacinglabel, tf.float32))

        # summary for tensorboard
        tf.summary.scalar('acc', self.tfvars.acc_combined)
        tf.summary.scalar('acc_label', self.tfvars.acc_variantlabel)
        tf.summary.scalar('acc_pacing', self.tfvars.acc_pacinglabel)
        tf.summary.scalar('loss', self.tfvars.loss)

    def startSession(self):
        """
        initialize tensorflow session and saver instance to save model weights
        """
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=0)
        self.sess.run(tf.train.get_or_create_global_step().initializer)

    def startSummary(self, summary_id):
        """
        initialize summary writers - we separate between train and test summaries and create different files for both cases
        :param summary_id: type of summary (e.g. train or test)
        """
        writer = tf.summary.FileWriter('%s-%s' % (self.get_model_save_name(self.config.summary_dir), summary_id), self.sess.graph)
        self.summary_writers[summary_id] = writer

    def initialize_variables(self):
        if not self.variables_initialized:
            self.sess.run(tf.global_variables_initializer())
            self.variables_initialized = True

    def add_trainop(self):
        if self.train_op is None:
            self.train_op = self.NNModel.optimize(self.tfvars.loss, self.tfvars.learning_rate)

    def train(self, lengths=None, val_length=None):
        """
        Actual training of neural network
        :param lengths:  lengths of time series used for training (e.g., we can train on histogram bins 0 to 60000
         and 30000 to 60000 to skip the first 30s)
        :param val_length: length used for validation step
        """
        logging.info('train routine')

        # use default lengths when not defined
        if lengths is None:
            lengths = [[0, -1]]

        repeat = len(lengths)

        if val_length is None:
            val_length = [0,-1]

        assert self.train_op is not None

        # get all summaries and join into one tensorflow op
        merged_summary = tf.summary.merge_all()

        # retrieve the datasets used for training and validation + initialize dataset iterator for later use
        # when using n lengths, we repeat every entry n times, so that during training we can train on n different lengths of the same entry
        dataset_train = self._get_tf_dataset(self.NNModel.get_trainbatchsize(), self.config.dataset_trainlabelfile,
                                             self.config.dataset_datapath, 'train', repeat=repeat)
        dataset_val = self._get_tf_dataset(self.NNModel.get_testbatchsize(), self.config.dataset_vallabelfile,
                                           self.config.dataset_datapath, 'val', shuffle=False)

        iterator_train = dataset_train.make_initializable_iterator()
        initializer_train = iterator_train.initializer
        handle_train = self.sess.run(iterator_train.string_handle())

        iterator_val = dataset_val.make_initializable_iterator()
        initializer_val = iterator_val.initializer
        handle_val = self.sess.run(iterator_val.string_handle())

        #
        train_op = self.train_op

        # initialize variables and save initial weights (maybe wanted for retraining)
        self.initialize_variables()
        self.save('initial')

        # variables used for early stopping
        min_val_loss = None
        max_val_acc = None
        not_improving_since = 0

        # actual training loop
        for epoch in range(self.NNModel.get_maxepoch()):
            # get current learning rate for this epoch
            lr = self.NNModel.learning_rate_schedule(epoch)

            logging.info('epoch: %d, lr: %f' % (epoch, lr))
            logging.info('train')

            # when training on n lengths, we aggregate the metrics per length
            # therefore we create n result helpers - which basically only average / sum up the defined metrics
            train_metrics =[]
            for i in range(repeat):
                train_metrics.append(ResultHelper(['loss', 'acc', 'acc_pacing', 'acc_variant', 'cm']))

            # reset train dataset iterator
            self.sess.run(initializer_train)
            try:
                while True:

                    # repeat for n lengths - note that we setup the dataset to repeat every entry n times
                    for i in range(repeat):
                        train_begin, train_end = lengths[i]
                        # actual training step
                        step_results = self._train_step(epoch, handle_train, train_op, lr, begin=train_begin,
                                                        end=train_end, summary=merged_summary, add_cm=True)
                        #aggregate metrics
                        train_metrics[i].add(step_results)
                        if i == 0:
                            #summarize only for first length
                            self.summarize('train', step_results['summary'])
            except tf.errors.OutOfRangeError:
                pass  # end of dataset

            # log metrics
            for i in range(repeat):
                train_begin, train_end = lengths[i]
                train_mean = train_metrics[i].mean()
                train_cm = train_metrics[i].sum()['cm']


                logging.info(
                    'summary, epoch: %d, [%d - %d to %d]: train_loss: %f, train_acc: %f, train_acc_variant: %f, train_acc_pacing: %f' % (
                        epoch, i, train_begin, train_end, train_mean['loss'], train_mean['acc'], train_mean['acc_variant'], train_mean['acc_pacing']))

                logging.info("confusion matrix\n%s" % str(train_cm))
                logging.info(cm_recall_precision_f1(train_cm))

            train_cm = train_metrics[0].sum()['cm']
            train_mean = train_metrics[0].mean()

            mean_summary = tf.Summary(value=[tf.Summary.Value(tag='mean_train_loss', simple_value=train_mean['loss']),
                                            tf.Summary.Value(tag='mean_train_acc', simple_value=train_mean['acc'])])
            self.summarize('train', mean_summary)

            ############################################################################################################

            # validation step
            logging.info('validate')

            val_metrics = ResultHelper(['loss', 'acc', 'acc_pacing', 'acc_variant', 'cm'])

            # reset val dataset iterator
            self.sess.run(initializer_val)
            try:
                # inference for every validation entry
                while True:
                    # note: only one length
                    val_begin, val_end = val_length
                    # actual inference / test step
                    step_results = self._test_step(handle_val, begin=val_begin, end=val_end,
                                                   summary=merged_summary, add_cm=True)
                    val_metrics.add(step_results)
            except tf.errors.OutOfRangeError:
                pass  # end of dataset

            # log metrics
            val_mean = val_metrics.mean()
            val_cm = val_metrics.sum()['cm']

            logging.info('val_loss: %f, val_acc: %f, val_acc_variant: %f, val_acc_pacing: %f' % (
                val_mean['loss'], val_mean['acc'], val_mean['acc_variant'], val_mean['acc_pacing']))
            logging.info("confusion matrix\n%s" % str(val_cm))
            logging.info(cm_recall_precision_f1(val_cm))

            mean_summary = tf.Summary(value=[tf.Summary.Value(tag='mean_val_loss', simple_value=val_mean['loss']),
                                            tf.Summary.Value(tag='mean_val_acc', simple_value=val_mean['acc'])])
            self.summarize('train', mean_summary)

            # log epoch metrics into json file

            json_metrics = {"epoch": epoch,
                              'metrics': {'val_loss': val_mean['loss'],
                                          'val_acc': val_mean['acc'],
                                          'val_acc_variant': val_mean['acc_variant'],
                                          'val_acc_pacing': val_mean['acc_pacing'],
                                          'val_cm': val_cm.tolist(),
                                          'train_loss': train_mean['loss'],
                                          'train_acc': train_mean['acc'],
                                          'train_acc_variant': train_mean['acc_variant'],
                                          'train_acc_pacing': train_mean['acc_pacing'],
                                          'train_cm': train_cm.tolist()}}

            with open("%s-train-metrics-epoch-%d.json" % (self.get_model_save_name(self.config.weight_dir), epoch), "w") as f:
                f.write(json.dumps(json_metrics))

            # Early Stopping
            val_loss = val_mean['loss']
            val_acc = val_mean['acc']

            if min_val_loss is None or val_loss < min_val_loss:
                min_val_loss = val_loss
                not_improving_since = 0
                # save weights of best loss achieved
                self.save('train-loss')
            else:
                not_improving_since += 1

            if max_val_acc is None or val_acc > max_val_acc:
                max_val_acc = val_acc
                not_improving_since = 0
                # save weights of best accuracy achieved
                self.save('train-acc')

            self.save('train-epoch-%d' % epoch)

            if not_improving_since > self.NNModel.get_earlystopping():
                logging.info('Early Stopping - No improvement after %d epochs' % self.NNModel.get_earlystopping())
                return

    def test(self, lengths=None):
        """
        Actual test of neural network - inference on test set and detailed information output to json file
        :param lengths: lengths used for inference (e.g., in 512ms steps increasing end for eval of flow length dependency)
        """
        logging.info('test routine')

        if lengths is None:
            repeat = 1
        else:
            repeat = len(lengths)

        # merge all summaries into single tensorflow op
        merged_summary = tf.summary.merge_all()

        # retrieve testing dataset and set up dataset iterator
        dataset_test = self._get_tf_dataset(self.NNModel.get_testbatchsize(), self.config.dataset_testlabelfile,
                                            self.config.dataset_datapath, 'test', shuffle=False, repeat=repeat, cache=False)

        iterator_test = dataset_test.make_initializable_iterator()
        initializer_test = iterator_test.initializer
        handle_test = self.sess.run(iterator_test.string_handle())

        # result helper aggregates metrics
        mh = ResultHelper(['loss', 'acc', 'acc_pacing', 'acc_variant', 'cm'])

        # list of results per entry
        json_results = []

        # reset dataset iterator
        self.sess.run(initializer_test)

        step_counter = 0
        try:
            #loop over dataset
            while True:
                logging.info("test_step %d" % step_counter)
                # repeat for every length defined
                for l in range(0, repeat):

                    if lengths is None:
                        end = -1
                        begin = 0
                    else:
                        end = lengths[l][1]
                        begin = lengths[l][0]

                    # inference step
                    step_results = self._test_step(handle_test, begin=begin, end=end, summary=merged_summary, add_cm=True, add_logits=True)

                    # last length used for summary and metrics
                    if l == repeat - 1:
                        mh.add(step_results)
                        self.summarize('test', step_results['summary'])

                    batch_size = len(step_results['logits'])

                    # iterate over every batch entry (so per pcap file) and aggregate logits for json output
                    for i in range(batch_size):
                        output_logits = step_results['logits'][i][-1].tolist()
                        output_filename = step_results['configfilename'][i].decode()
                        output_btln = int(step_results['afterbtln'][i])
                        output_opts = datasethelper.get_opts(output_filename)
#                        logging.info(output_opts)
#                        logging.info(output_logits)
#                        logging.info(step_results['real_pacing'][i])
#                        logging.info(step_results['pred_pacing'][i][-1])
#                        logging.info(step_results['real_variant'][i])
#                        logging.info(step_results['pred_variant'][i][-1])
                        if l == 0:
                            entry = {'logits': [(begin, end, output_logits)],
                                     'opts': output_opts,
                                     'btln': output_btln,
                                     'filename': output_filename
                                    }
                            json_results.append(entry)
                        else:
                            j = -batch_size + i
                            assert json_results[j]['filename'] == output_filename
                            json_results[j]['logits'].append((begin, end, output_logits))

                step_counter += 1
        except tf.errors.OutOfRangeError:
            pass  # end of dataset


        #log metrics
        mean = mh.mean()

        logging.info('test_loss: %f, test_acc: %f, test_acc_variant: %f, test_acc_pacing: %f' % (
            mean['loss'], mean['acc'], mean['acc_variant'], mean['acc_pacing']))
        logging.info("confusion matrix\n%s" % str(mh.sum()['cm']))
        logging.info(cm_recall_precision_f1(mh.sum()['cm']))

        mean_summary = tf.Summary(value=[tf.Summary.Value(tag='mean_test_loss', simple_value=mean['loss']),
                                         tf.Summary.Value(tag='mean_test_acc', simple_value=mean['acc'])])
        self.summarize('test', mean_summary)

        #save json with test logits
        model_save_name = self.get_model_save_name(self.config.weight_dir)

        with open("%s-test.json" % model_save_name, "w") as f:
            hostname = socket.gethostname()
            f.write(json.dumps({'entries': json_results, 'model_name': self.name, 'model_save_name': model_save_name, 'hostname': hostname}))

        pass


    def _train_step(self, epoch, handle_train, train_op, learning_rate, summary, begin=0, end=-1, add_logits=False,
                    add_cm=False, acc_op=None, acc_it=0):
        """
        Actual train step of neural network
        :param epoch: training epoch
        :param handle_train: dataset handle for training dataset
        :param train_op: training tensorflow op (e.g. adam)
        :param learning_rate: learning rate to be used
        :param summary: tensorflow merged summary
        :param begin: start of histogram slice
        :param end: end of histogram slice
        :param add_logits: whether logits are returned
        :param add_cm: whether confusion matrix is returned
        :param acc_op: reserved
        :param acc_it: reserved
        :return:
        """

        # run helper allows to link op and result
        rh = RunHelper(self.sess)

        rh.add(train_op, '_train_op')
        rh.add(summary, 'summary')
        rh.add(self.tfvars.loss, 'loss')
        rh.add(self.tfvars.acc_combined, 'acc')
        rh.add(self.tfvars.acc_variantlabel, 'acc_variant')
        rh.add(self.tfvars.acc_pacinglabel, 'acc_pacing')

        if add_logits:
            rh.add(self.tfvars.logits, 'logits')

        if add_cm:
            rh.add(self.tfvars.confusion_matrix, 'cm')

        rh.feed(self.dataset_handle, handle_train)
        rh.feed(self.tfvars.entry_begin, begin)
        rh.feed(self.tfvars.entry_end, end)
        rh.feed(self.tfvars.is_train, True) # we are training, so is_train should be true!
        rh.feed(self.tfvars.learning_rate, learning_rate)

        results = rh.run()

        return results

    def _test_step(self, handle_val, summary, begin=0, end=-1, add_logits=False, add_cm=False, read_config=False):
        """
        Actual inference step of neural network
        :param handle_val: validation
        :param summary: tensorflow merged summary
        :param begin: start of histogram slice
        :param end: end of histogram slice
        :param add_logits: whether logits are returned
        :param add_cm: whether confusion matrix is returned
        :param read_config: whether also config files are read and contents returned
        :return:
        """
        rh = RunHelper(self.sess)

        rh.add(self.tfvars.loss, 'loss')
        rh.add(summary, 'summary')
        rh.add(self.tfvars.acc_combined, 'acc')
        rh.add(self.tfvars.acc_variantlabel, 'acc_variant')
        rh.add(self.tfvars.acc_pacinglabel, 'acc_pacing')
        # whether entry was record before or after bottleneck link
        rh.add(self.dataset_entries.afterbtlnlabel, 'afterbtln')
        rh.add(self.dataset_entries.variantlabel, 'real_variant')
        rh.add(self.dataset_entries.pacinglabel, 'real_pacing')
        rh.add(self.tfvars.pred_pacinglabel, 'pred_pacing')
        rh.add(self.tfvars.pred_variantlabel, 'pred_variant')
        # config filename (json file) for entry
        rh.add(self.dataset_entries.configfilename, 'configfilename')

        if add_logits:
            rh.add(self.tfvars.logits, 'logits')

        if add_cm:
            rh.add(self.tfvars.confusion_matrix, 'cm')

        rh.feed(self.dataset_handle, handle_val)
        rh.feed(self.tfvars.entry_begin, begin)
        rh.feed(self.tfvars.entry_end, end)
        rh.feed(self.tfvars.is_train, False)

        results = rh.run()

        if read_config:
            configs = []
            # for every entry, read configfile and add content through datasethelper
            for configfilename in results['configfilename']:
                configs.append(datasethelper.get_opts(configfilename))
            results['config'] = configs

        return results

    def _get_tf_dataset(self, batch_size, labelfile, datapath, name, shuffle=True, repeat=1, cycle=1, cache=True):
        """
        get tensorflow dataset via deepcci dataset helper
        :param batch_size: batch size of dataset
        :param labelfile: file, which contains labels and links to data file
        :param datapath: path, in which datafiles will be found
        :param name: mainly used for cache file creation
        :param shuffle: whether data should be shuffled
        :param repeat: whether entries should be repeated
        :param cycle: distance between repeated entries (can be used when accumulating gradients)
        :return:
        """

        def _repeat_tf_dataset(dataset, repeat, cyclelength=1):
            """
            repeats entries in dataset
            :param dataset: actual dataset
            :param repeat: how often to repeat entries
            :param cyclelength: distance between repetitions
            :return: dataset with repetitions ( if repeat > 1 otherwise same as input)
            """
            assert type(repeat) == int and repeat > 0
            if repeat > 1:
                dataset = dataset.interleave(lambda s, c, d, v, p, b: tf.data.Dataset.from_tensors(
                    (s, c, d, v, p, b)).repeat(repeat), cycle_length=cyclelength, block_length=1)
            return dataset

        # get basic dataset from datasethelper
        dataset = datasethelper.get_tf_dataset(rootpath=self.config.dataset_rootpath,
                                               labelfile=labelfile,
                                               parallel=self.config.dataset_parallel_readers,
                                               datapath=datapath,
                                               configpath=self.config.dataset_configpath)

        # caching of datasets for faster retrieval due to lower IO and deserialization overhead
        # we splitted every entry into a separate file causing IO / FS overhead due to many files
        # caching in RAM is of course faster, but requires massive amounts of RAM
        # therefore also caching in a file is possible
        # (no FS overhead due to separate files + lower deserialization overhead
        # as saved in intermediate representation)

        if cache:
            if self.config.dataset_cache_ram:
                dataset = dataset.cache()

            elif self.config.dataset_cache_file:
                cache_name = '%s.cache' % name

                pattern = '%s*' % cache_name
                files = glob.glob(pattern)
                # we remove previous cache files - maybe dataset changed
                logging.info('remove cache files of %s' % cache_name)
                logging.info(files)
                for file in files:
                    os.remove(file)

                dataset = dataset.cache(cache_name)

        # shuffle dataset when wanted
        if shuffle:
            dataset = dataset.shuffle(self.config.dataset_shufflesize * batch_size)

        # prefetch for concurrent retrieval while training
        dataset = dataset.prefetch(buffer_size=self.config.dataset_prefetchsize * batch_size)
        # batching m entries into one tensor
        dataset = dataset.batch(batch_size)
        # repeat batches when wanted
        dataset = _repeat_tf_dataset(dataset, repeat, cyclelength=cycle)
        return dataset

    def set_extraid(self, id):
        """
        extra id for filenames
        :param id: ID
        """
        self.extraid = id

    def set_expected_dimension(self, dim):
        """
        expected dimensions / features of dataset - helps tensorflow in setting up weight matrices
        :param dim: expected amount of features (typically 1 for us, as we use only the arrival histogram bin)
        """
        self.config.dataset_expected_dimensions = dim

    def summarize(self, summary_id, summary):
        """
        add summary results to summary file
        :param summary_id: summary id ( e.g., train or test)
        :param summary: summary content
        """
        if summary_id in self.summary_writers:
            self.summary_writers[summary_id].add_summary(summary, tf.train.get_or_create_global_step().eval(self.sess))
            self.summary_writers[summary_id].flush()

    def get_model_save_name(self, dir):
        """
        aggregate model instance IDs into a filename
        :param dir: rootdir
        :return: model_save_name
        """
        prefixname = self.name
        prefixname += '-%s' % str(self.uid)

        if self.extraid is not None:
            prefixname += '-%s' % self.extraid

        if self.file_loaded is not None:
            # whether training operation existed prior to loading weight file (important when e.g. adam parameters are loaded or freshly generated)
            filename, trainop_existed = self.file_loaded
            prefixname += '-%s-%d' % (filename.replace(".","-").replace("/","-"), 1 if trainop_existed else 0)

        prefixname += '-%s' % self.time.strftime('%Y-%m-%d_%H:%M')
        model_save_name = '%s/%s' % (dir, prefixname)

        return model_save_name

    def save(self, postfix):
        """
        save model weights
        :param postfix: postfix for file
        """
        model_save_name = self.get_model_save_name(self.config.weight_dir)
        self.saver.save(self.sess, '%s-%s' % (model_save_name, postfix))


    def load(self, filename):
        """
        load model weights
        :param filename: filename of weightfile
        """
        self.saver.restore(self.sess, filename)
        # whether training operation existed beforehand (important when e.g. adam parameters are loaded or freshly generated)
        if self.train_op:
            self.file_loaded = (filename, True)
        else:
            self.file_loaded = (filename, False)

    def set_dataset_caching(self, cache):
        self.config.dataset_cache_file = cache
        pass

    def set_dataset_ramcaching(self, ramcache):
        self.config.dataset_cache_ram = ramcache
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--model', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--id', type=str)
    args = parser.parse_args()

    cache = True
    ramcache = False
    lstm_type = None
    # machine.settings allows to define machine settings (such as whether ram caching or using tensorflow LSTM impl. for CPU inference)
    # uses json syntax
    if os.path.exists('machine.settings'):
        with open('machine.settings') as f:
            settings = json.loads(f.read())
            if 'cache' in settings:
                cache = settings['cache']
            if 'ramcache' in settings:
                ramcache = settings['ramcache']
            if 'lstm_type' in settings:
                lstm_type = settings['lstm_type']
    else:
        logging.info("missing machine.settings file - using defaults")

    if not args.config:
        logging.info("please select config file with --config")
        return

    if not os.path.exists(args.config):
        logging.info("config file does not exist")
        return

    with open(args.config) as f:
        conf = json.loads(f.read())

    nn = NNModel(config=conf)
    if lstm_type:
        nn.set_lstmtype(lstm_type)

    # use config file name as model name
    d = DeepcciTF(args.config.replace('.','-').replace("/","-"), nn)

    #set up datasets
    d.set_dataset_caching(cache)
    d.set_dataset_ramcaching(ramcache)

    d.set_dataset_datapath(conf['data_path'])
    d.set_dataset_testlabelfile(conf['test_file'])
    d.set_dataset_trainlabelfile(conf['train_file'])
    d.set_dataset_vallabelfile(conf['val_file'])
    d.set_expected_dimension(conf['expected_dimensions'])

    if args.id:
        d.set_extraid(args.id)
    
    if args.train:
        d.add_trainop()
    
    d.startSession()

    if args.model:
        d.load(args.model)

    if args.train:
        d.startSummary('train')
        d.train(lengths=conf['train_lengths'], val_length=conf['val_length'])

    if args.test:
        if args.train is not None or args.model is not None:
            d.startSummary('test')
            d.test(lengths=conf['test_lengths'])
        else:
            logging.info('Please train or give model weights for testing')

    if not args.test and not args.train:
        logging.info("use --test to initiate testing or --train to start training")

if __name__ == '__main__':
    main()