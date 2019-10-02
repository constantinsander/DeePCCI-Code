import os

import numpy as np
import tensorflow as tf

from utils.utils import JsonSerialization


# deepcci dataset parsing to create tensorflow dataset API dataset

def get_expected_types():
    """
    expected types of this dataset
    :return: tuple of tensorflow types
    """
    return (tf.float32, tf.string, tf.string, tf.int32, tf.int32, tf.int32)


def get_expected_output_shapes():
    """
    output shape of this dataset
    :return: tuple of dimensions
    """
    return ([None, None, None], None, None, None, None, None)


def get_tf_dataset(rootpath, labelfile, parallel, datapath, configpath):
    """
    get tensorflow dataset from human-readable deepcci dataset
    :param rootpath: dataset rootpath where labels and datapath lie
    :param labelfile: labelfilename
    :param parallel: amount of concurrent threads
    :param datapath: path to be used for dataset entries
    :param configpath: path to be used when searching for configfilename
    :return:
    """

    def _prepare_line(line):
        """
        parsing labelfile line
        :param line: current line from label file
        :return: parsed results
        """

        # read variant, pacing and after bottleneck label (so whether entry recorded before or after bottleneck) +
        # entry configfilename and entry datafilename form csv file
        variantlabel, pacinglabel, afterbtlnlabel, configfilename, datafilename = line.split(",")
        configfilename = configfilename.strip()
        datafilename = datafilename.strip()

        # join pathes
        datafilename = os.path.join(rootpath, datapath, datafilename)
        configfilename = os.path.join(configpath, configfilename)

        variantlabel = int(variantlabel)
        pacinglabel = int(pacinglabel)
        afterbtlnlabel = int(afterbtlnlabel)

        return configfilename, datafilename, variantlabel, pacinglabel, afterbtlnlabel

    # reading labelfile through textlinedataset
    dataset = tf.data.TextLineDataset(os.path.join(rootpath, labelfile))

    def _parse_info_tf(line):
        """
        tensorflow wrapper for parse_info wrapping prepare_line
        :param line: current text line
        :return: tensorflow op of parse_info / configfilename, datafilename, variantlabel, pacinglabel, afterbtlnlabel
        """

        def _parse_info_py(line):
            """
            wrap parse_line to return tensorflow compatible types
            :param line: text line
            :return: parsed results - tensorflow compatible / configfilename, datafilename, variantlabel, pacinglabel, afterbtlnlabel
            """
            configfilename, datafilename, variantlabel, pacinglabel, afterbtlnlabel = _prepare_line(line.decode())
            return configfilename, datafilename, np.int32(variantlabel), np.int32(pacinglabel), np.int32(afterbtlnlabel)

        return tf.py_func(_parse_info_py, [line], [tf.string, tf.string, tf.int32, tf.int32, tf.int32])

    # map parse function onto textline dataset
    dataset = dataset.map(_parse_info_tf,
                          num_parallel_calls=parallel)  # TODO calls py_func - is not concurrent - however, pyfunc rather small

    def _parse_tf(configfilename, datafilename, variantlabel, pacinglabel, afterbtlnlabel):
        """
        read entry data file
        :param configfilename: parsed configfilename from label file
        :param datafilename: parsed entry data filename
        :param variantlabel: variant label
        :param pacinglabel: pacing label
        :param afterbtlnlabel: bottleneck label
        :return: read data file sequence, configfilename, datafilename, variantlabel, pacinglabel, afterbtlnlabel
        """

        # using tensorflow ops for concurrency as files rather large around 120kB)

        # read file content (is csv file)
        text = tf.read_file(datafilename)
        # split per line - assume one feature per line
        lines = tf.string_split([text], "\n").values
        # expand dimensions to lines x 1 / features x 1
        lines = tf.expand_dims(lines, -1)

        # split per comma in csv file ( -> lines x time step / features x time step)
        sequence = tf.map_fn(lambda x: tf.string_split(x, ",").values, lines)

        # convert strings from file to ints
        sequence = tf.string_to_number(sequence) / 10.0  # normalize by maximum (slow start 10 packets)

        # swap features x time step to time step x features
        sequence = tf.transpose(sequence, [1, 0])

        return sequence, configfilename, datafilename, variantlabel, pacinglabel, afterbtlnlabel

    dataset = dataset.map(_parse_tf, num_parallel_calls=parallel)
    return dataset


def get_opts(configfilename):
    """
    read config file and parse it (including our own json deserialization)
    :param configfilename:
    :return:
    """
    # tensorflow results and python3 require decoding
    if type(configfilename) == bytes:
        configfilename = configfilename.decode()

    # read config file and deserialize
    with open("%s.json" % configfilename) as f:
        opts = JsonSerialization.loads(f.read())

    return opts
