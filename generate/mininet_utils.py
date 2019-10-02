import math
import os
import re
import shutil

from mininet.link import Link, Intf
from mininet.net import Mininet
from mininet.topo import Topo
from mininet.util import waitListening, output, debug, error, info


class TCLinkTBF(Link):
    """
    Link with symmetric TC interfaces configured via opts
    - splits left and right settings
    """

    def __init__(self, node1, node2, port1=None, port2=None,
                 intfName1=None, intfName2=None,
                 addr1=None, addr2=None, **params):
        params1 = {}
        params2 = {}
        if "params1" in params:
            params1 = params["params1"]
        if "params2" in params:
            params2 = params["params2"]

        Link.__init__(self, node1, node2, port1=port1, port2=port2,
                      intfName1=intfName1, intfName2=intfName2,
                      cls1=TCIntfTBF,
                      cls2=TCIntfTBF,
                      addr1=addr1, addr2=addr2,
                      params1=params1,
                      params2=params2)


# adapted from original mininet source
#    our changes:
#      - adapted token bucket filter to use queue size
#      - token bucket filter setup as leaky bucket

class TCIntfTBF(Intf):
    """
    Interface customized by tc (traffic control) utility
    Allows specification of bandwidth limits (various methods)
    as well as delay, loss and max queue length
    """

    # The parameters we use seem to work reasonably up to 1 Gb/sec
    # For higher data rates, we will probably need to change them.
    bwParamMax = 1000

    def bwCmds(self, bw=None, speedup=0, max_queue_size=None, enable_ecn=False, enable_red=False):
        "Return tc commands to set bandwidth"

        cmds, parent = [], ' root '

        if bw and (bw < 0 or bw > self.bwParamMax):
            error('Bandwidth limit', bw, 'is outside supported range 0..%d'
                  % self.bwParamMax, '- ignoring\n')
        elif bw is not None:

            if max_queue_size:
                # use max_queue_size in token bucket filter queue
                print(bw)
                netem = False
                if netem:
                    cmds += ['%s qdisc add dev %s root handle 5: netem ' +
                             'rate %fmbit limit %d' %
                             (bw, int(math.ceil(max_queue_size / 1540.)))]
                else:
                    cmds += ['%s qdisc add dev %s root handle 5: tbf ' +
                             'rate %fmbit burst 1540 limit %d' %
                             (bw, int(math.ceil(max_queue_size / 1540.) * 1540))]
            else:
                # cmds += ['%s qdisc add dev %s root handle 5: tbf ' +
                #         'rate %fmbit burst 1522' %
                #         (bw)]
                pass
            parent = ' parent 5:1 '

            # ECN or RED
            if enable_ecn:
                cmds += ['%s qdisc add dev %s' + parent +
                         'handle 6: red limit 1000000 ' +
                         'min 30000 max 35000 avpkt 1500 ' +
                         'burst 20 ' +
                         'bandwidth %fmbit probability 1 ecn' % bw]
                parent = ' parent 6: '
            elif enable_red:
                cmds += ['%s qdisc add dev %s' + parent +
                         'handle 6: red limit 1000000 ' +
                         'min 30000 max 35000 avpkt 1500 ' +
                         'burst 20 ' +
                         'bandwidth %fmbit probability 1' % bw]
                parent = ' parent 6: '
        return cmds, parent

    @staticmethod
    def fqCmds(parent, fq=None):
        "Internal method: return tc commands for delay and loss"
        cmds = []
        if fq:
            cmds = ["%s qdisc add dev %s " + parent + "handle 20: fq"]
            parent = ' parent 20:1 '
        return cmds, parent

    @staticmethod
    def delayCmds(parent, delay=None, jitter=None,
                  loss=None, max_queue_size=None):
        "Internal method: return tc commands for delay and loss"
        cmds = []
        if delay and delay < 0:
            error('Negative delay', delay, '\n')
        elif jitter and jitter < 0:
            error('Negative jitter', jitter, '\n')
        elif loss and (loss < 0 or loss > 100):
            error('Bad loss percentage', loss, '%%\n')
        else:
            # Delay/jitter/loss/max queue size
            netemargs = '%s%s%s%s' % (
                'delay %s ' % delay if delay is not None else '',
                '%s ' % jitter if jitter is not None else '',
                'loss %d ' % loss if loss is not None else '',
                'limit %d' % math.ceil(max_queue_size / 1540.) if max_queue_size is not None
                else '')
            if netemargs:
                cmds = ['%s qdisc add dev %s ' + parent +
                        ' handle 10: netem ' +
                        netemargs]
                parent = ' parent 10:1 '
        return cmds, parent

    def tc(self, cmd, tc='tc'):
        "Execute tc command for our interface"
        c = cmd % (tc, self)  # Add in tc command and our name
        debug(" *** executing command: %s\n" % c)
        return self.cmd(c)

    def config(self, bw=None, delay=None, jitter=None, loss=None,
               gro=False, txo=True, rxo=True,
               speedup=0, enable_ecn=False, enable_red=False,
               max_queue_size=None, limit=None, fq=None, instantiation=None, **params):
        """Configure the port and set its properties.
           bw: bandwidth in b/s (e.g. '10m')
           delay: transmit delay (e.g. '1ms' )
           jitter: jitter (e.g. '1ms')
           loss: loss (e.g. '1%' )
           gro: enable GRO (False)
           txo: enable transmit checksum offload (True)
           rxo: enable receive checksum offload (True)
           speedup: experimental switch-side bw option
           use_hfsc: use HFSC scheduling
           use_tbf: use TBF scheduling
           latency_ms: TBF latency parameter
           enable_ecn: enable ECN (False)
           enable_red: enable RED (False)
           max_queue_size: queue limit parameter for tbf"""

        # Support old names for parameters
        gro = not params.pop('disable_gro', not gro)

        result = Intf.config(self, **params)

        def on(isOn):
            "Helper method: bool -> 'on'/'off'"
            return 'on' if isOn else 'off'

        # Set offload parameters with ethool
        self.cmd('ethtool -K', self,
                 'gro', on(gro),
                 'tx', on(txo),
                 'rx', on(rxo))

        if instantiation is not None:
            instantiation(self)

        self.cmd('ethtool -K', self, 'tso', 'off', 'gso', 'off')

        # Optimization: return if nothing else to configure
        # Question: what happens if we want to reset things?
        if (bw is None and not delay and not loss
                and max_queue_size is None and fq is None):
            return

        # Clear existing configuration
        tcoutput = self.tc('%s qdisc show dev %s')
        if "priomap" not in tcoutput and "noqueue" not in tcoutput:
            cmds = ['%s qdisc del dev %s root']
        else:
            cmds = []

        # Bandwidth limits via various methods
        bwcmds, parent = self.bwCmds(bw=bw, speedup=speedup,
                                     enable_ecn=enable_ecn,
                                     enable_red=enable_red,
                                     max_queue_size=max_queue_size)
        cmds += bwcmds

        # Delay/jitter/loss/limit using netem
        # limit to set netem queue size - unused for us
        delaycmds, parent = self.delayCmds(delay=delay, jitter=jitter,
                                           loss=loss,
                                           max_queue_size=limit,
                                           parent=parent)
        cmds += delaycmds

        fqcmds, parent = self.fqCmds(parent=parent, fq=fq)

        cmds += fqcmds

        # Ugly but functional: display configuration info
        stuff = ((['%.2fMbit' % bw] if bw is not None else []) +
                 (['%s delay' % delay] if delay is not None else []) +
                 (['%s jitter' % jitter] if jitter is not None else []) +
                 (['%d%% loss' % loss] if loss is not None else []) +
                 (['fq'] if fq is not None else []) +
                 (['ECN'] if enable_ecn else ['RED']
                 if enable_red else []))
        info('(' + ' '.join(stuff) + ') ')

        # Execute all the commands in our node
        debug("at map stage w/cmds: %s\n" % cmds)
        tcoutputs = [self.tc(cmd) for cmd in cmds]
        for output in tcoutputs:
            if output != '':
                error("*** Error: %s" % output)
        debug("cmds:", cmds, '\n')
        debug("outputs:", tcoutputs, '\n')
        result['tcoutputs'] = tcoutputs
        result['parent'] = parent

        # Set offload parameters with ethtool
        error(self.tc('%s qdisc show dev %s'))

        return result


# adapted from Mininet source
# added congestion control support for iperf and own traffic gen
class MininetCong(Mininet):
    congModes = ['cubic', 'bic', 'bbr', 'reno', 'westwood', 'htcp', 'hstcp', 'hybla', 'vegas', 'nv', 'scalable', 'lp',
                 'veno', 'yeah', 'illinois', 'dctcp', 'cdr']

    def iperf(self, hosts=None, l4Type='TCP', udpBw='10M', fmt=None,
              seconds=5, port=5001, cong='cubic'):
        """Run iperf between two hosts.
           hosts: list of hosts; if None, uses first and last hosts
           l4Type: string, one of [ TCP, UDP ]
           udpBw: bandwidth target for UDP test
           fmt: iperf format argument if any
           seconds: iperf time to transmit
           port: iperf port
           returns: two-element array of [ server, client ] speeds
           note: send() is buffered, so client rate can be much higher than
           the actual transmission rate; on an unloaded system, server
           rate should be much closer to the actual receive rate"""
        hosts = hosts or [self.hosts[0], self.hosts[-1]]
        # CONFIG_TCP_CONG_BIC = m
        # CONFIG_TCP_CONG_CUBIC = y
        # CONFIG_TCP_CONG_WESTWOOD = m
        # CONFIG_TCP_CONG_HTCP = m
        # CONFIG_TCP_CONG_HSTCP = m
        # CONFIG_TCP_CONG_HYBLA = m
        # CONFIG_TCP_CONG_VEGAS = m
        # CONFIG_TCP_CONG_NV = m
        # CONFIG_TCP_CONG_SCALABLE = m
        # CONFIG_TCP_CONG_LP = m
        # CONFIG_TCP_CONG_VENO = m
        # CONFIG_TCP_CONG_YEAH = m
        # CONFIG_TCP_CONG_ILLINOIS = m
        # CONFIG_TCP_CONG_DCTCP = m
        # CONFIG_TCP_CONG_CDG = m
        # CONFIG_TCP_CONG_BBR = m
        # congModes = ['CUBIC', 'BIC', 'BBR', 'WESTWOOD', 'HTCP', 'HSTCP', 'HYBLA', 'VEGAS', 'NV', 'SCALABLE', 'LP', 'VENO', 'YEAH', 'ILLINOIS', 'DCTCP', 'CDR']
        assert len(hosts) == 2
        assert cong in self.congModes
        if cong == "hstcp":
            cong = "highspeed"

        client, server = hosts
        output('*** Iperf: testing', l4Type, 'bandwidth between',
               client, 'and', server, '\n')
        iperfArgs = 'iperf -m -p %d ' % port
        bwArgs = ''
        if l4Type == 'UDP':
            iperfArgs += '-u '
            bwArgs = '-b ' + udpBw + ' '
        elif l4Type != 'TCP':
            raise Exception('Unexpected l4 type: %s' % l4Type)
        if fmt:
            iperfArgs += '-f %s ' % fmt
        server.sendCmd(iperfArgs + '-s')
        if l4Type == 'TCP':
            if not waitListening(client, server.IP(), port):
                raise Exception('Could not connect to iperf on port %d'
                                % port)
        cliout = client.cmd(iperfArgs + '-t %d -c ' % seconds +
                            server.IP() + ' ' + bwArgs + ' -Z %s' % cong)
        debug('Client output: %s\n' % cliout)
        servout = ''
        # We want the last *b/sec from the iperf server output
        # for TCP, there are two of them because of waitListening
        count = 2 if l4Type == 'TCP' else 1
        while len(re.findall('/sec', servout)) < count:
            servout += server.monitor(timeoutms=5000)
        server.sendInt()
        servout += server.waitOutput()
        debug('Server output: %s\n' % servout)
        result = [self._parseIperf(servout), self._parseIperf(cliout)]
        if l4Type == 'UDP':
            result.insert(0, udpBw)
        output('*** Results: %s\n' % result)
        return result

    def ownStartServer(self, server, seconds):
        server.sendCmd('./trafficgen -s -t %d' % seconds)

    def ownStartClient(self, server, client, seconds, cong, pacing):
        if cong == "hstcp":
            cong = "highspeed"
        pacingargs = "-P" if pacing else ""
        client.sendCmd('./trafficgen -t %d -h %s -c %s %s' % (seconds, server.IP(), cong, pacingargs))

    def ownStopServer(self, host):
        return host.waitOutput()

    def ownStopClient(self, host):
        return host.waitOutput()

    def ownTestConnection(self, server, client):
        return waitListening(client, server.IP(), 5001)

    @staticmethod
    def _parseOwn(output):
        if not "time is over" in output:
            raise Exception('"time is over" not found in trafficgen output')
        return True

    def iperfStartServer(self, server, l4Type='TCP', udpBw='10M', fmt=None, port=5001, timeout=None):
        iperfArgs = 'iperf -m -p %d ' % port
        bwArgs = ''
        if l4Type == 'UDP':
            iperfArgs += '-u '
            bwArgs = '-b ' + udpBw + ' '
        elif l4Type != 'TCP':
            raise Exception('Unexpected l4 type: %s' % l4Type)
        if fmt:
            iperfArgs += '-f %s ' % fmt
        if timeout:
            iperfArgs += '-t %d ' % timeout
        server.sendCmd(iperfArgs + '-s')

    @staticmethod
    def _parseIperf(iperfOutput):
        r = r'([\d\.]+ \w+/sec)'
        m = re.findall(r, iperfOutput)
        if m:
            return m[-1]
        else:
            raise Exception('could not parse iperf output: ' + iperfOutput)

    def iperfStopServer(self, host):
        host.sendInt()
        return host.waitOutput()

    def iperfStopClient(self, host):
        return host.waitOutput()

    def iperfTestConnection(self, server, client, l4Type='TCP', port=5001):
        if l4Type == 'TCP':
            return waitListening(client, server.IP(), port)
        return True

    def iperfStartClient(self, server, client, l4Type='TCP', udpBw='10M', fmt=None,
                         seconds=5, port=5001, cong='cubic'):
        iperfArgs = 'iperf -m -p %d ' % port
        bwArgs = ''
        if l4Type == 'UDP':
            iperfArgs += '-u '
            bwArgs = '-b ' + udpBw + ' '
        elif l4Type != 'TCP':
            raise Exception('Unexpected l4 type: %s' % l4Type)
        if fmt:
            iperfArgs += '-f %s ' % fmt
        client.sendCmd(iperfArgs + '-t %d -c ' % seconds +
                       server.IP() + ' ' + bwArgs + ' -Z %s' % cong)


class DynamicTopo(Topo):
    # create topo from json topology
    class linkobj:
        """
        callback based link obj to get link names on instantiation
        """

        def __init__(self):
            self.obj = None

        def set(self, name):
            self.obj = name

        def get(self):
            return self.obj

        def __call__(self, *args, **kwargs):
            return self.obj

    def add_bottleneck(self, bw, delay, queue, jitter):
        """
        build bottleneck by emulating bottleneck part by part (bandwidth restriction + delay)
        allows to record traffic at different points of bottleneck
        :param bw: bandwidth of bottleneck
        :param delay: delay of bottleneck
        :param queue: queue size of bottleneck
        :param jitter: jitter of delay of bottleneck (unused)
        :return: left side of link, right side of link, linux link left, linux link right
        """

        #                            delay     bwlim
        #    <------    <------    <------    <------    <------
        # s0-rec     s1         s2         s3         s4     rec-s5
        #    ------>    ------>    ------>    ------>    ------>
        #                bwlim      delay

        # add virtual switches
        s0 = self.addSwitch('s%d' % self.i)
        self.i += 1
        s1 = self.addSwitch('s%d' % self.i)
        self.i += 1
        s2 = self.addSwitch('s%d' % self.i)
        self.i += 1
        s3 = self.addSwitch('s%d' % self.i)
        self.i += 1
        s4 = self.addSwitch('s%d' % self.i)
        self.i += 1
        s5 = self.addSwitch('s%d' % self.i)
        self.i += 1

        # use 10 BDP for netem delay queue (as bandwidth limited by TBF, 1 BDP should suffice)
        limit = 10 * math.ceil(delay.toSeconds() * bw.toBit() / 8.)

        jitters = None
        if jitter is not None:
            jitters = '%dms' % (jitter.toMiliseconds())

        # print("connect %s - %s" % (s2, s3))
        self.addLink(s2, s3,
                     params1=dict(limit=limit, delay='%dms' % (delay.toMiliseconds()), jitter=jitters),
                     params2=dict(limit=limit, delay='%dms' % (delay.toMiliseconds()), jitter=jitters)
                     )

        # bandwidth limiting link in right direction
        # print("connect %s - %s" % (s1, s2))
        self.addLink(s1, s2,
                     params1=dict(max_queue_size=math.ceil(queue.toBit() / 8.),
                                  bw=bw.toFullMbit())
                     )

        # bandwidth limiting link in left direction
        # print("connect %s - %s" % (s3, s4))
        self.addLink(s3, s4,
                     params2=dict(max_queue_size=math.ceil(queue.toBit() / 8.),
                                  bw=bw.toFullMbit())
                     )

        # links are not directly instantiated with addLink, but first when the mininet is built
        # we therefore set a callback for instantiation for the links, we are interested in,
        # in which the interface name is saved durign building
        left = DynamicTopo.linkobj()
        right = DynamicTopo.linkobj()

        # print("connect %s - %s" % (s0, s1))
        self.addLink(s0, s1, params1={'instantiation': left.set})

        # print("connect %s - %s" % (s4, s5))
        self.addLink(s4, s5, params2={'instantiation': right.set})

        return s0, s5, ((left, s0), (right, s5))

    def buildTopo(self, topo):
        """
        recreate topology from topology json
        :param topo: json topo
        :return: senders, receivers, recordinglinks, senderrecord, devcon
        """
        self.h += 1
        devices, connections, recordings, gencon = topo

        recordinglinks = []
        receivers = []
        senders = []
        senderrecord = []

        # build devices used in network
        dictofdevices = {}
        for device in devices:
            deepcopy = {}
            deepcopy.update(device)
            device = deepcopy
            if device['type'] == 'bottleneck':
                jitter = None
                if 'jitter' in device:
                    jitter = device['jitter']
                device['__internal__'] = self.add_bottleneck(bw=device['bw'], delay=device['delay'],
                                                             queue=device['queue'], jitter=jitter)

            if device['type'] == 'sender':
                device['__internal__'] = self.addHost('h1-%s-%d' % (device['id'], self.h))
                if device.get('rec', False):
                    senderrecord.append(device['__internal__'])
                else:
                    senders.append(device['__internal__'])

            if device['type'] == 'receiver':
                device['__internal__'] = self.addHost('h2-%s-%d' % (device['id'], self.h))
                receivers.append(device['__internal__'])

            if device['type'] == 'switch':
                device['__internal__'] = self.addSwitch('s-%s-%d' % (device['id'], self.i))
                self.i += 1
            dictofdevices[device['id']] = device

        # connect devices
        for deviceaid, devicebid in connections:
            a = None
            b = None
            linkparams = {}

            if dictofdevices[deviceaid]['type'] == 'bottleneck':
                a = dictofdevices[deviceaid]['__internal__'][1]
            elif dictofdevices[deviceaid]['type'] == 'sender' and dictofdevices[deviceaid]['pacing']:
                a = dictofdevices[deviceaid]['__internal__']
                linkparams['params1'] = dict(fq=True)
            else:
                a = dictofdevices[deviceaid]['__internal__']

            if dictofdevices[devicebid]['type'] == 'bottleneck':
                b = dictofdevices[devicebid]['__internal__'][0]
            elif dictofdevices[devicebid]['type'] == 'sender' and dictofdevices[devicebid]['pacing']:
                a = dictofdevices[devicebid]['__internal__']
                linkparams['params2'] = dict(fq=True)
            else:
                b = dictofdevices[devicebid]['__internal__']

            self.addLink(a, b, **linkparams)

        # create recording
        for btln, where, name in recordings:
            a = dictofdevices[btln]['__internal__']
            if where == "left":
                linkintf, switch = a[2][0]
            elif where == "right":
                linkintf, switch = a[2][1]
            else:
                assert False
            recordinglinks.append((linkintf, switch, name))

        # create connections between sender and receivers
        devcon = []
        for a, b in gencon:
            assert dictofdevices[a]['type'] == 'sender'
            assert dictofdevices[b]['type'] == 'receiver'
            devcon.append((dictofdevices[a]['__internal__'], dictofdevices[b]['__internal__'], dictofdevices[a]['cong'],
                           dictofdevices[a]['pacing']))

        return senders, receivers, recordinglinks, senderrecord, devcon

    def getSenders(self):
        return self.senders

    def getRecordsenders(self):
        return self.recordsenders

    def getReceivers(self):
        return self.receivers

    def getRecordingLinks(self):
        return self.recordinglinks

    def getRecordDevCons(self):
        # devcon : device connections
        return self.recorddevcons

    def build(self, params):
        self.senders = []
        self.receivers = []
        self.recordinglinks = []
        self.recordsenders = []
        self.recorddevcons = []

        self.i = 0
        self.h = 0

        for param in params:
            topo = param['topo']
            senders, receivers, recordinglinks, senderrecord, devcon = self.buildTopo(topo)
            self.senders.append(senders)
            self.receivers.append(receivers)
            self.recordinglinks.append(recordinglinks)
            self.recordsenders.append(senderrecord)
            self.recorddevcons.append(devcon)


class PcapRecorder:
    """
    tcpdump abstraction on mininet devices
    """

    def __init__(self, device, name, interface, limit_ip, shouldname=None):
        """

        :param device: virtual switch on which to execute tcpdump command
        :param name: filename (incl path)
        :param interface: linux interface name
        :param limit_ip: ip on which to limit (currently unused)
        :param shouldname: filename (incl path) after recording finished
        """
        self.device = device
        self.name = name
        self.interface = interface
        self.limit_ip = limit_ip
        if shouldname is None:
            shouldname = name
        self.shouldname = shouldname

    def start(self):
        #        self.device.cmd('tcpdump "host %s" -n -s 100 -i %s -w %s &' % (self.limit_ip, self.interface, self.name))
        self.device.cmd('tcpdump -n -s 100 -i %s -w %s &' % (self.interface, self.name))

    def stop(self):
        self.device.cmd('kill %tcpdump')
        if self.shouldname != self.name:
            shutil.copy(self.name, self.shouldname)
            os.remove(self.name)
