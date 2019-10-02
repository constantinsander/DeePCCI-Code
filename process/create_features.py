import functools
import multiprocessing as mp
import os.path
import sys
from glob import glob

sys.path.append("..")

from utils.utils import JsonSerialization as json
from utils.utils import mbit

names_to_variant_idx = {'bbr': 0, 'cubic': 1, 'reno': 2, 'cubic-pacing': 1, 'reno-pacing': 2}
names_to_pacing_idx = {'bbr': 0, 'cubic': 0, 'reno': 0, 'cubic-pacing': 1, 'reno-pacing': 1}


def sort_if_necessary(p):
    """
    sorts arrival of packets when wrong order detected
    :param p: packet arrival
    :return: sorted arrival
    """
    oldt = None
    sorting_necessary = False
    for t, l, seq in p:
        if oldt is not None:
            if oldt > t:
                # previous timestamp was higher than current timestamp?!
                print((oldt, t, "warning"))
                sorting_necessary = True
        oldt = t
    if sorting_necessary:
        p = sorted(p, key=lambda x: x[0])
    return p


def get_src_ip(opts):
    """
    get ip of sender for filtering
    :param opts: config file content
    :return: IP of sender
    """
    return opts['ip']


def get_right_flow(d, settings):
    """
    extract sender flow from flows
    :param d: flows
    :param settings: config file content
    :return: flow to server / from sender, flow to client / to sender, filename or None when not found
    """
    for flow in d:
        if len(flow) == 0:
            continue
        (toserver, toclient), filename, flowid = flow
        opts = settings
        # is sender flow?
        if flowid[0][0] != get_src_ip(opts):
            continue
        # less than 10 exchanged packets? something was wrong
        if len(toclient) < 10 or len(toserver) < 10:
            continue
        return toserver, toclient, filename
    return None


def get_histogram(ps, steps=0.1, end=2.0):
    """
    compute histogram from packet arrivals
    :param ps: packet arrivals
    :param steps: bin size
    :param end: histogram size
    :return: histogram
    """
    size = int(end / steps)
    histograms = []

    for packets in ps:
        histogram = [0] * size
        t0 = None  # histogram time origin
        for t, l, seq in packets:
            if t0 is None:
                t0 = t
            if t - t0 >= end:
                break
            # find according histogram bine
            i = int((t - t0) / steps)
            histogram[i] += 1
        histograms.append(histogram)

    return histograms


def _csv(n, hist):
    """
    write list as csv into file n
    :param n: filename
    :param hist: list to write (e.g. histogram)
    """
    with open(n, "w") as f:
        f.write('\n'.join([','.join(map(str, h)) for h in hist]))


def mp_create_hist(outdir, filenames_and_config):
    """
    create histograms for files
    :param outdir: dir to output histograms to
    :param filenames_and_config: preparsed config and datafilename
    :return: csv entry line
    """
    (config, before, after, k) = filenames_and_config
    cong = config['config']['cong']

    # get variant and pacing label from name
    variant_idx = names_to_variant_idx[cong]
    pacing_idx = names_to_pacing_idx[cong]

    # check whether files already exist - if yes, skip histogram creation
    file_exists = os.path.isfile("%s/1ms/%s_before.csv" % (outdir, k)) and os.path.isfile(
        "%s/1ms/%s_before.csv" % (outdir, k))
    if file_exists:
        return [(variant_idx, pacing_idx, 0, "%s" % k, "%s_before.csv" % k),
                (variant_idx, pacing_idx, 1, "%s" % k, "%s_after.csv" % k)]

    # read packet arrival jsons
    try:
        before_file = open(before, 'r')
        after_file = open(after, 'r')
        before_data = json.loads(before_file.read())
        after_data = json.loads(after_file.read())
    except Exception as e:
        print(e, "problem with", (before, after))
        return None

    # extract flows from packet arrival
    before_flow_data = get_right_flow(before_data, config)
    after_flow_data = get_right_flow(after_data, config)

    if before_flow_data is None or after_flow_data is None:
        print("problem parsing", (before, after))
        return None

    # extract flow directions
    before_flow, before_flow_back, _ = before_flow_data
    after_flow, after_flow_back, _ = after_flow_data

    # extract timestamp, length and sequence number
    packets_before = [(packet['t'], packet['l'], packet['seq']) for packet in before_flow if packet['l'] > 0]
    packets_after = [(packet['t'], packet['l'], packet['seq']) for packet in after_flow if packet['l'] > 0]

    # sort if wrong order
    packets_before = sort_if_necessary(packets_before)
    packets_after = sort_if_necessary(packets_after)

    # create histograms
    hist_before = get_histogram([packets_before], steps=0.001, end=60.0)
    hist_after = get_histogram([packets_after], steps=0.001, end=60.0)

    # write histograms to csv files
    _csv("%s/1ms/%s_before.csv" % (outdir, k), hist_before)
    _csv("%s/1ms/%s_after.csv" % (outdir, k), hist_after)

    # return two csv label file lines (before and after bottleneck)
    return [(variant_idx, pacing_idx, 0, "%s" % k, "%s_before.csv" % k),
            (variant_idx, pacing_idx, 1, "%s" % k, "%s_after.csv" % k)]


def get_filenames_and_config():
    """
    find all files, check consistency and read config
    :return: filenames and configs as list
    """

    # where to search for preprocessed data
    dir = "../data/preprocessed/"

    # find all files
    files = glob(dir + "*.pcap.json")
    files = [f[len(dir):] for f in files]

    # consistency check: do pcaps before and after bottleneck exist?
    names = {}
    for f in files:
        name = None
        add = 0
        if f.endswith("+before.pcap.json"):
            name = f[:-len("+before.pcap.json")]
            add = 1
        elif f.endswith("+after.pcap.json"):
            name = f[:-len("+after.pcap.json")]
            add = 2
        if name:
            if name not in names:
                names[name] = 0
            names[name] += add

    filenames = []
    for k, v in names.items():
        if v < 3:
            # one of both pcaps is missing
            print("missing files for %s" % k)
        else:
            if os.path.exists("../data/jsons/%s.json" % k):
                # everything alright
                beforepcap = "%s%s+before.pcap.json" % (dir, k)
                afterpcap = "%s%s+after.pcap.json" % (dir, k)
                with open("../data/jsons/%s.json" % k) as f:
                    config = json.loads(f.read())
                # add filenames, config and general id
                filenames.append((config, beforepcap, afterpcap, k))
            else:
                # configfile missing
                print("missing files for %s" % k)

    return filenames


def create_hists_json():
    """
    main entry point for histogram creation
    """

    # get all files and their config
    filenames_and_config = get_filenames_and_config()

    # setup multiprocessing
    p = mp.Pool(20)

    # should multiprocessing be used? hard to debug
    multip = True

    # fuse function call with first parameter (outdir) being set
    call = functools.partial(mp_create_hist, "../data/processed")

    # decide if parallel multiprocessing or iterative, single process
    if multip:
        ret = p.map(call, filenames_and_config)
    else:
        ret = []
        for f in filenames_and_config:
            print(f)
            ret.append(call(f))

    zipped = zip(filenames_and_config, ret)

    # create label csv files (train, test, val).csv
    create_label_files(zipped)

    # check where histogram creation failed for final eval
    ret = [r for r in ret if r is not None]
    print(len(ret), "from", len(filenames_and_config))

    # write all labels in all.csv
    ret = [x for r in ret for x in r]
    _csv("%s/all.csv" % ("../data/processed"), ret)


def create_label_files(zipped):
    """
    sort files depending on their bandwidth into different dataset label files
    :param zipped: config and csv lines zipped per entry (before and after bottleneck)
    """
    train_files = []
    test_files = []
    val_files = []
    cross_files = []

    # define sorting constraints - here bandwidths
    test_bws = [mbit(2), mbit(10), mbit(25), mbit(50)]
    val_bws = [mbit(4), mbit(30)]

    # sort into different label files
    for filename_and_config, csv_ret in zipped:
        if csv_ret is None:
            # :(
            continue

        config, before, after, k = filename_and_config

        topoid = config['topoid']
        bw = config['config']['bw']

        for csv_line in csv_ret:
            if not topoid in ['singlehost', 'multihost']:
                # probably only crosstraffic network
                cross_files.append(csv_line)
            if bw in test_bws:
                test_files.append(csv_line)
            elif bw in val_bws:
                val_files.append(csv_line)
            else:
                train_files.append(csv_line)

    _csv("%s/train.csv" % ("../data/processed"), train_files)
    _csv("%s/test.csv" % ("../data/processed"), test_files)
    _csv("%s/val.csv" % ("../data/processed"), val_files)
    _csv("%s/cross.csv" % ("../data/processed"), cross_files)


if __name__ == '__main__':
    create_hists_json()
