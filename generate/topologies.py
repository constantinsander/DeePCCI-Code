from utils.utils import bit


# topology definitions

def adapt_bdp_factor(topology):
    """
    calculate queue size from bandwidth and delay and bdp_factor
    :param topology: topology entry
    :return: adapted topology entry with queue size
    """
    topology['queue'] = bit(topology['bw'].toBit() * topology['delay'].toSeconds() * topology['BDP'])
    return topology


def get_single_host_topology(cong, bdp_factor, delay, bw):
    """
    topology for single host network
    :param cong: recorded sender congestion control
    :param bdp_factor: bdp_factor bottleneck
    :param delay: delay bottleneck
    :param bw: bandwidth bottleneck
    :return: topology definition
    """

    # parse congestion and pacing from name - if reno-pacing then pacing = true, if reno pacing = false - cong = reno for both
    c = cong.split("-")
    cong = c[0]
    pacing = len(c) > 1

    return ([{'type': 'sender', 'cong': cong, 'pacing': pacing, 'id': "sen11", 'rec': True},
             {'type': 'receiver', 'id': "rec11"},
             adapt_bdp_factor({'type': 'bottleneck', 'BDP': bdp_factor, 'delay': delay, 'bw': bw, 'id': "btln1"}),
             {'type': 'switch', 'id': "sw1"},
             {'type': 'switch', 'id': "sw2"},
             {'type': 'switch', 'id': "sw3"},
             {'type': 'switch', 'id': "sw4"},
             ], [("sen11", "sw1"),
                 ("rec11", "sw2"),
                 ("sw1", "btln1"), ("btln1", "sw2"), ],
            [("btln1", "left", "before"), ("btln1", "right", "after")],
            [('sen11', 'rec11')])


def get_multi_host_topology(cong, cong1, cong2, bdp_factor, delay, bw):
    """
    topology for multi host network
    :param cong: recorded sender congestion control
    :param cong1: congestion control background sender 1
    :param cong2: congestion control background sender 2
    :param bdp_factor: bdp_factor bottleneck
    :param delay: delay bottleneck
    :param bw: bandwidth bottleneck
    :return: topology definition
    """

    # parse congestion and pacing from name - if reno-pacing then pacing = true, if reno pacing = false - cong = reno for both
    def congsplit(cong):
        c = cong.split("-")
        cong = c[0]
        pacing = len(c) > 1
        return cong, pacing

    cong1, pacing1 = congsplit(cong1)
    cong2, pacing2 = congsplit(cong2)
    cong, pacing = congsplit(cong)

    return ([{'type': 'sender', 'cong': cong1, 'pacing': pacing1, 'id': "sen11"},
             {'type': 'sender', 'cong': cong2, 'pacing': pacing2, 'id': "sen12"},
             {'type': 'sender', 'cong': cong, 'pacing': pacing, 'id': "sen13", 'rec': True},
             {'type': 'receiver', 'id': "rec11"},
             {'type': 'receiver', 'id': "rec12"},
             {'type': 'receiver', 'id': "rec13"},
             adapt_bdp_factor({'type': 'bottleneck', 'BDP': bdp_factor, 'delay': delay, 'bw': bw, 'id': "btln1"}),
             {'type': 'switch', 'id': "sw1"},
             {'type': 'switch', 'id': "sw2"},
             {'type': 'switch', 'id': "sw3"},
             {'type': 'switch', 'id': "sw4"},
             ], [("sen11", "sw1"), ("sen12", "sw1"), ("sen13", "sw1"),
                 ("rec11", "sw2"), ("rec12", "sw2"), ("rec13", "sw2"),
                 ("sw1", "btln1"), ("btln1", "sw2"), ],
            [("btln1", "left", "before"), ("btln1", "right", "after")],
            [('sen11', 'rec11'), ('sen12', 'rec12'), ('sen13', 'rec13')])


def get_crosstraffic_topology(cong, bw, link1_bdp, link2_bdp, link3_bdp, link4_bdp, link5_bdp,
                              link1_delay, link2_delay, link3_delay, link4_delay, link5_delay):
    """
    topology for crosstraffic network
    :param cong: recorded sender congestion control
    :param bw: bandwidth of main links
    :param link1_bdp: bdp_factor link 1
    :param link2_bdp: bdp_factor link 2
    :param link3_bdp: bdp_factor link 3
    :param link4_bdp: bdp_factor link 4
    :param link5_bdp: bdp_factor link 5
    :param link1_delay: delay link 1
    :param link2_delay: delay link 2
    :param link3_delay: delay link 3
    :param link4_delay: delay link 4
    :param link5_delay: delay link 5
    :return: topology definition
    """

    # parse congestion and pacing from name - if reno-pacing then pacing = true, if reno pacing = false - cong = reno for both
    c = cong.split("-")
    cong = c[0]
    pacing = len(c) > 1

    return ([{'type': 'sender', 'cong': 'bbr', 'pacing': False, 'id': "sen11"},
             {'type': 'sender', 'cong': 'reno', 'pacing': False, 'id': "sen12"},
             {'type': 'sender', 'cong': 'cubic', 'pacing': False, 'id': "sen13"},
             {'type': 'sender', 'cong': cong, 'pacing': pacing, 'id': "sen14", 'rec': True},
             {'type': 'sender', 'cong': 'bbr', 'pacing': False, 'id': "sen21"},
             {'type': 'sender', 'cong': 'cubic', 'pacing': False, 'id': "sen22"},
             {'type': 'sender', 'cong': 'reno', 'pacing': False, 'id': "sen23"},
             {'type': 'receiver', 'id': "rec11"},
             {'type': 'receiver', 'id': "rec12"},
             {'type': 'receiver', 'id': "rec13"},
             {'type': 'receiver', 'id': "rec14"},
             {'type': 'receiver', 'id': "rec21"},
             {'type': 'receiver', 'id': "rec22"},
             {'type': 'receiver', 'id': "rec23"},
             adapt_bdp_factor({'type': 'bottleneck', 'BDP': link1_bdp,
                               'delay': link1_delay, 'bw': bw, 'id': "btln1"}),
             adapt_bdp_factor({'type': 'bottleneck', 'BDP': link2_bdp,
                               'delay': link2_delay, 'bw': bw / 2, 'id': "btln2"}),
             adapt_bdp_factor({'type': 'bottleneck', 'BDP': link3_bdp,
                               'delay': link3_delay, 'bw': bw, 'id': "btln3"}),
             adapt_bdp_factor({'type': 'bottleneck', 'BDP': link4_bdp,
                               'delay': link4_delay, 'bw': bw / 2, 'id': "btln4"}),
             adapt_bdp_factor({'type': 'bottleneck', 'BDP': link5_bdp,
                               'delay': link5_delay, 'bw': bw, 'id': "btln5"}),
             {'type': 'switch', 'id': "sw1"},
             {'type': 'switch', 'id': "sw2"},
             {'type': 'switch', 'id': "sw3"},
             {'type': 'switch', 'id': "sw4"},
             ], [("sen11", "sw1"), ("sen12", "sw1"), ("sen13", "sw1"), ("sen14", "sw1"),
                 ("rec11", "sw2"), ("rec12", "sw2"), ("rec13", "sw2"), ("rec14", "sw2"),
                 ("sen21", "sw3"), ("sen22", "sw3"), ("sen23", "sw3"),
                 ("rec21", "sw4"), ("rec22", "sw4"), ("rec23", "sw4"),
                 ("btln1", "btln3"), ("btln2", "btln3"), ("btln3", "btln4"), ("btln3", "btln5"),
                 ("sw1", "btln1"), ("sw3", "btln2"), ("btln5", "sw2"), ("btln4", "sw4")],
            [("btln3", "left", "before"), ("btln3", "right", "after")],
            [('sen11', 'rec11'), ('sen12', 'rec12'), ('sen13', 'rec13'), ('sen14', 'rec14'),
             ('sen21', 'rec21'), ('sen22', 'rec22'), ('sen23', 'rec23')])
