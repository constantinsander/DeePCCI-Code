import itertools
import sys

sys.path.append("..")

from utils.utils import params_gen, milliseconds, mbit, dict_to_filename
from utils.utils import JsonSerialization as json
from generate.topologies import get_crosstraffic_topology, get_single_host_topology, get_multi_host_topology

ms = milliseconds

# directory where to save param files
dir = "params/"


def get_single_host_params():
    """
    list of all params used for single host network
    :return: list of all params
    """
    return params_gen({
        'delay': [ms(x) for x in
                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90,
                   100, 150, 200, 250, 300]],
        'bdp_factor': [0.5, 1, 2, 5, 10],
        'bw': [mbit(x) for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]],
        'cong': ['bbr', 'cubic', 'reno', 'reno-pacing', 'cubic-pacing']
    })


def get_multi_host_params():
    """
    list of all params used for multi host network
    :return: list of all params
    """
    return params_gen({
        'delay': [ms(x) for x in [1, 2, 5, 10, 20, 50]],
        'bdp_factor': [1, 5, 10],
        'bw': [mbit(x) for x in [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 100]],
        'cong': ['bbr', 'cubic', 'reno', 'reno-pacing', 'cubic-pacing'],
        'cong12': itertools.combinations_with_replacement(['bbr', 'cubic', 'reno', 'reno-pacing', 'cubic-pacing'], 2)
    })


def get_crosstraffic_params():
    """
    list of all params used for crosstraffic host network
    :return: list of all params
    """
    return params_gen({
        'link1_delay': [ms(10)],
        'link1_bdp': [1, 5, 10],
        'link2_delay': [ms(10)],
        'link2_bdp': [1, 5, 10],
        'link3_delay': [ms(10)],
        'link3_bdp': [1, 5, 10],
        'link4_delay': [ms(10)],
        'link4_bdp': [1, 5, 10],
        'link5_delay': [ms(10)],
        'link5_bdp': [1, 5, 10],
        'cong': ['bbr', 'cubic', 'reno', 'reno-pacing', 'cubic-pacing'],
        'bw': [mbit(50)]
    })


def create_single_host_runs():
    """
    combine single host network params and topology to config file
    """
    params = get_single_host_params()
    for param in params:
        filename = "single," + dict_to_filename(param)
        run = {}
        run['pcapName'] = "single"
        run['time'] = 60
        run['topoid'] = 'singlehost'
        run['topo'] = get_single_host_topology(**param)
        run['config'] = param

        with open(dir + filename + ".json", "w") as f:
            f.write(json.dumps(run))


def create_multi_host_runs():
    """
    combine multi host network params and topology to config file
    """
    params = get_multi_host_params()
    for param in params:
        cong1, cong2 = param['cong12']
        del param['cong12']
        param['cong1'] = cong1
        param['cong2'] = cong2
        filename = "multi," + dict_to_filename(param)
        run = {}
        run['pcapName'] = "multi"
        run['time'] = 60
        run['topoid'] = 'multihost'
        run['topo'] = get_multi_host_topology(**param)
        run['config'] = param

        with open(dir + filename + ".json", "w") as f:
            f.write(json.dumps(run))


def create_crosstraffic_runs():
    """
    combine crosstraffic network params and topology to config file
    """
    params = get_crosstraffic_params()
    for param in params:
        filename = "cross," + dict_to_filename(param)
        run = {}
        run['pcapName'] = "cross"
        run['time'] = 60
        run['topoid'] = 'crosstraffic'
        run['topo'] = get_crosstraffic_topology(**param)
        run['config'] = param

        with open(dir + filename + ".json", "w") as f:
            f.write(json.dumps(run))


def create_runs():
    """
    main entry
    """
    create_crosstraffic_runs()
    create_multi_host_runs()
    create_single_host_runs()


if __name__ == '__main__':
    create_runs()
