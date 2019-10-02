import numpy as np


class DatasetEntry:
    """
    map dataset entries with names for overview
    """

    def __init__(self, sequence, configfilename, datafilename, variantlabel, pacinglabel, afterbtlnlabel):
        self.sequence = sequence
        self.configfilename = configfilename
        self.datafilename = datafilename
        self.variantlabel = variantlabel
        self.pacinglabel = pacinglabel
        self.afterbtlnlabel = afterbtlnlabel


class RunHelper:
    """
    RunHelper allows to link tensorflow ops with names which are also used for the results.
    normally session.run returns values for ops in same order as ops are given as parameter.
    RunHelper links op and results by name.
    """

    def __init__(self, session):
        """
        :param session: tensorflow session
        """
        self.session = session
        self.ops = []
        self.names = []
        self.feed_dict = {}

    def add(self, op, name):
        """
        add tensorflow op to be executed
        :param op: tensorflow op
        :param name: name of op - used as dictionary key for op's result
        """
        self.ops.append(op)
        self.names.append(name)

    def feed(self, placeholder, value):
        """
        also feed placeholder values for session.run
        :param placeholder: tensorflow placeholder
        :param value: placeholder feed value
        :return:
        """
        self.feed_dict[placeholder] = value

    def run(self):
        """
        Run ops while feeding placeholders as setup using feed
        :return: dictionary of op results (key is op name used in add)
        """
        results = self.session.run(self.ops, feed_dict=self.feed_dict)
        results = dict(zip(self.names, results))
        return results


def cm_recall_precision_f1(cm):
    """
    debug confusion matrix to precision, recall and f1 scores
    :param cm: confusion matrix
    :return: list of strings per class containing scores in textual form for debugging
    """
    out = []
    n, _ = cm.shape
    for i in range(n):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fn - fp

        if tp == 0:
            recall = 0
            precision = 0
            f1 = 0
        else:
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)

            f1 = 2 * precision * recall / (precision + recall)

        out.append("class %d, precision %f, recall %f, f1 %f" % (i, precision, recall, f1))
    return "metrics: %s" % "\n".join(out)


class ResultHelper(object):
    """
    Avoid saving all results separately, but compute mean by summing results and dividing at the end
    """

    def __init__(self, keys):
        """
        :param keys: which keys to aggregate
        """
        self.keys = keys
        self.data = None
        self.length = 0

    def add(self, results):
        """
        add new metric results
        :param results: results from train or test step
        """
        if not self.data:
            self.data = {}
            for key in self.keys:
                self.data[key] = results[key]
        else:
            for key in self.keys:
                self.data[key] += results[key]
        self.length += 1

    def mean(self):
        """
        compute mean of results
        :return: average / mean of added metrics
        """
        assert self.length != 0
        out = []
        for key in self.keys:
            out.append((key, self.data[key] / self.length))
        return dict(out)

    def sum(self):
        """
        compute sum of metrics
        :return: sum of added metrics
        """
        out = []
        for key in self.keys:
            out.append((key, self.data[key]))
        return dict(out)
