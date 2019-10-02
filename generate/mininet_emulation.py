import os
import shutil
import sys
from time import sleep

from mininet.clean import cleanup
from mininet.log import lg
from mininet.node import OVSBridge

sys.path.append("../")
import uuid

import logging

from generate.mininet_utils import MininetCong, PcapRecorder, TCLinkTBF, DynamicTopo
from utils.utils import JsonSerialization as json


# mininet emulation glue code

def do(datadir, jsondir, tmpdir, runs):
    """

    :param datadir: directory where to save pcaps
    :param jsondir: directory where to save jsons
    :param tmpdir:  directory where to save pcaps temporarily
    :param runs: list of topology information
    """
    runtime_params = []

    # it is possible to have multiple topologies in parallel (untested, we only used one)
    for run in runs:
        runtime_params.append({})
        runtime_params[-1]['time'] = run['time']
        runtime_params[-1]['opts'] = run
        ok = False

        while True:
            # find unused filenames
            uniqueid = uuid.uuid4().hex
            runtime_params[-1]['uniquename'] = run['pcapName'] + '+' + uniqueid

            logfilename = jsondir + runtime_params[-1]['uniquename'] + ".log"
            templogfilename = tmpdir + '/' + runtime_params[-1]['uniquename'] + ".log"
            runtime_params[-1]['templogfilename'] = templogfilename
            runtime_params[-1]['logfilename'] = logfilename

            runtime_params[-1]['jsonfilename'] = jsondir + runtime_params[-1]['uniquename'] + ".json"

            if not os._exists(logfilename):
                break

        # log into file
        runtime_params[-1]['filehandler'] = logging.FileHandler(templogfilename)

        runtime_params[-1]['filehandler'].setLevel(logging.INFO)
        lg.addHandler(runtime_params[-1]['filehandler'])

    retry = 0
    while True:
        retry += 1

        # setup topology from params
        topo = DynamicTopo(runs)
        net = MininetCong(topo=topo,
                          link=TCLinkTBF,
                          controller=None,  # OVSController,
                          switch=OVSBridge)
        # setup and build network
        net.start()
        net.waitConnected()

        for runtime in runtime_params:
            lg.output("%s\n" % runtime['opts'])

        run_senders = topo.getSenders()
        run_recordsenders = topo.getRecordsenders()
        run_receivers = topo.getReceivers()
        run_recorddevcons = topo.getRecordDevCons()
        # print(run_recorddevcons)
        run_recordlinks = topo.getRecordingLinks()

        switch_by_name = {s.name: s for s in net.switches}
        hosts_by_name = {h.name: h for h in net.hosts}

        run_senders = [[hosts_by_name[sender] for sender in senders] for senders in run_senders]
        run_receivers = [[hosts_by_name[receiver] for receiver in receivers] for receivers in run_receivers]
        run_recordsenders = [[hosts_by_name[recordsender] for recordsender in recordsenders] for recordsenders in
                             run_recordsenders]
        run_recorddevcons = [[(hosts_by_name[a], hosts_by_name[b], c, p) for a, b, c, p in devcons] for devcons in
                             run_recorddevcons]

        recs = []

        # set up recordings
        for senders, recordlinks, runtime, recordsenders in zip(run_senders, run_recordlinks, runtime_params,
                                                                run_recordsenders):
            last_sender = recordsenders[0]
            assert len(recordsenders) == 1
            # switches = [(s0, "s0"), (s1, "s1"), (s4, "s4"), (s5, "s5")]
            # edgeswitches = [(switch_by_name[first_switch], "s0"), (switch_by_name[last_switch], "s5")]

            runtime['opts']['ip'] = last_sender.IP()

            for link, switchname, name in recordlinks:
                print(link())
                switch = switch_by_name[switchname]
                filename = runtime['uniquename'] + "+" + name + ".pcap"
                recs.append(
                    PcapRecorder(switch, tmpdir + '/' + filename, link().name, last_sender.IP(), datadir + filename))

        # try up to 40 times to ping the connected hots
        for i in range(40):
            l = 0
            for senders, receivers, recordsenders in zip(run_senders, run_receivers, run_recordsenders):
                all_hosts = senders + recordsenders + receivers
                l += net.ping(all_hosts)

            if l == 0:
                break

        sleep(2)

        # start trafficgen server
        for receiver, runtime in zip(run_receivers, runtime_params):
            for h2 in receiver:
                lg.output("start own server %s\n" % h2.name)
                # h1 is client sender, h2 is server receiver
                # data is sent from h1 to h2
                net.ownStartServer(h2, seconds=runtime['time'] + 10)
                # h2.cmd("timeout 20 nc -l -p 5001 > /dev/null &")

        sleep(2)

        # start test is server is listening
        con = True
        for recorddevcon, runtime in zip(run_recorddevcons, runtime_params):
            for (h1, h2, c, p) in recorddevcon:
                lg.output("test connection between server %s and client %s\n" % (h2.name, h1.name))
                con = net.ownTestConnection(h2, h1)
                if not con:
                    break
        if not con:
            lg.output("connection failed\n")
            with open("errors.txt", "a+") as f:
                f.write("connection failed\n")
            for receiver in run_receivers:
                for h2 in receiver:
                    lg.output("stop own server %s\n" % h2.name)
                    # h1 is client, h2 is server
                    # data is sent from h1 to h2
                    net.ownStopServer(h2)
            try:
                net.stop()
            except:
                pass
            cleanup()
            if retry <= 3:
                continue
            else:
                lg.output("3 retries failed\n")
                with open("errors.txt", "a+") as f:
                    f.write("3 retries failed\n")
                break

        lg.output("run generation\n")

        # start client (sender) of background connections
        i = 0
        for recordsenders, devcon, runtime in zip(run_recordsenders, run_recorddevcons, runtime_params):
            # for (h1, h2) in zip(sender, receiver):
            for (h1, h2, cong, pacing) in devcon:
                if h1 in recordsenders:
                    continue
                net.ownStartClient(h2, h1, seconds=runtime['time'] + 2 + 2, cong=cong, pacing=pacing)
                # h1.cmd("timeout 15 nc 10.0.0.1 5001 & ")
                i += 1

        # start tcpdump recording
        for rec in recs:
            rec.start()

        sleep(2)

        # start client (sender) which should be recorded
        for recordsenders, devcon, runtime in zip(run_recordsenders, run_recorddevcons, runtime_params):
            # for (h1, h2) in zip(sender, receiver):
            for (h1, h2, cong, pacing) in devcon:
                if not h1 in recordsenders:
                    continue
                net.ownStartClient(h2, h1, seconds=runtime['time'], cong=cong, pacing=pacing)
                # h1.cmd("timeout 10 dd if=/dev/zero | nc 10.0.0.2 5001")
                #        net.iperf((h1, h2), seconds=5, cong=cong[-1])

        sleep(max([runtime['time'] for runtime in runtime_params]) + 2)

        # stop recording
        for rec in recs:
            rec.stop()

        # stop client and server and check whether succesful
        try:
            for sender, receiver, recordsender, runtime in zip(run_senders, run_receivers, run_recordsenders,
                                                               runtime_params):
                for h1 in sender + recordsender:
                    # h1 is client, h2 is server
                    lg.output("stop %s\n" % h1.name)
                    net._parseOwn(net.ownStopClient(h1))
                for h2 in receiver:
                    lg.output("stop %s\n" % h2.name)
                    net._parseOwn(net.ownStopServer(h2))

        except Exception as e:
            lg.output("stopping hosts failed\n")
            with open("errors.txt", "a+") as f:
                f.write("stopping hosts failed " + str(e) + "\n")
            for sender, receiver, recordsender, runtime in zip(run_senders, run_receivers, run_recordsenders,
                                                               runtime_params):
                for h1 in sender + recordsender:
                    # h1 is client, h2 is server
                    lg.output("stop %s\n" % h1.name)
                    net.ownStopClient(h1)
                for h2 in receiver:
                    lg.output("stop %s\n" % h2.name)
                    net.ownStopServer(h2)
            cleanup()
            if retry <= 3:
                continue
            else:
                lg.output("3 retries failed\n")
                with open("errors.txt", "a+") as f:
                    f.write("3 retries failed\n")
                break

        net.stop()
        ok = True
        break

    print("remove handler\n")
    for runtime in runtime_params:
        with open(runtime['jsonfilename'], 'w') as f:
            f.write(json.dumps(runtime['opts']))
            f.flush()
        lg.removeHandler(runtime['filehandler'])
        shutil.move(runtime['templogfilename'], runtime['logfilename'])
    cleanup()
    print("done\n")
    return ok
