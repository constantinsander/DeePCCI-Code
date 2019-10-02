#include <stdio.h>
#include <unistd.h>
#include <tins/tins.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <stddef.h>
#include <glob.h>
#include <map>
#include <bitset>
#include <queue>
#include <sstream>
#include <semaphore.h>
#include <fstream>
#include <sys/types.h> /* key_t, sem_t, pid_t      */
#include <sys/ipc.h>
#include <sys/sem.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>

//number of concurrent threads
#define NUM_THREADS 12

struct tcp_packet {
    uint32_t seq;
    uint32_t ack;
    int flags;
    time_t sec;
    suseconds_t usec;
    int l;
};

using namespace Tins;
using flow_ident = std::tuple<std::string,std::string,int,int>;

//https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c
inline bool exists (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0); 
}

void analyze(std::string& filename, std::map<flow_ident, std::pair<int,int>>& flowstate, std::map<std::pair<flow_ident, int>, std::pair<std::vector<tcp_packet>,std::vector<tcp_packet>>>& flowdata, std::vector<std::pair<flow_ident, int>>& floworder) {
    flowstate.clear();
    floworder.clear();
    flowdata.clear();

    std::stringstream output_filename;
    std::string base_filename = filename.substr(filename.find_last_of("/\\") + 1);
    output_filename << "../data/preprocessed/" << base_filename << ".json";

    bool file_exists = exists(output_filename.str());

    if(file_exists) {
            std::cerr << "outfile exists already" << std::endl;
            return;
    }

    std::cerr << "analyze " << filename << std::endl;

    //Tins PCAP processing
    FileSniffer sniffer(filename);

    sniffer.sniff_loop([&](const Packet& packet) {

    const PDU& pdu = *packet.pdu();
    const IP* ip = pdu.find_pdu<IP>();
    const TCP* tcp = pdu.find_pdu<TCP>();

    //ugly TCP state parsing - works for our generated data
    //Tins own state parsing was too slow
    if(ip != NULL && tcp != NULL) {
        flow_ident fromflow;
	    flow_ident toflow;
//      std::cerr << "src: " << ip->src_addr() << ", dst: " << ip->dst_addr() << ", srcport: " << tcp->sport() << ", dport: " << tcp->dport() << std::endl;
        std::get<0>(fromflow) = ip->src_addr().to_string();
        std::get<1>(fromflow) = ip->dst_addr().to_string();
        std::get<2>(fromflow) = tcp->sport();
        std::get<3>(fromflow) = tcp->dport();

        std::get<0>(toflow) = ip->dst_addr().to_string();
        std::get<1>(toflow) = ip->src_addr().to_string();
        std::get<2>(toflow) = tcp->dport();
        std::get<3>(toflow) = tcp->sport();

        if(tcp->flags() == TCP::SYN) {
            if(flowstate.find(fromflow) == flowstate.end()) {
                std::pair<int,int> state;
                state.first = 0;
                state.second = 0;
                flowstate[fromflow] = state;
            }

            std::pair<int,int>& state = flowstate[fromflow];

            if(state.first != 0) {
//                std::cerr << "invalid SYN - connection already exists" << std::endl;
            }
            else {
                state.first = 1;
                state.second++;
                floworder.push_back(std::make_pair(fromflow, state.second));
                std::vector<tcp_packet> v;
                std::vector<tcp_packet> v2;
                flowdata[std::make_pair(fromflow, state.second)] = std::make_pair(v,v2);
            }

            tcp_packet t;
            t.seq = tcp->seq();
            t.ack = tcp->ack_seq();
            t.flags = tcp->flags();
            t.sec = packet.timestamp().seconds();
            t.usec = packet.timestamp().microseconds();
            t.l = 0;

            if(tcp->inner_pdu() != 0)
              t.l = ip->tot_len() - 20 - 4*tcp->data_offset();

            flowdata[std::make_pair(fromflow, state.second)].first.push_back(t);
        }
        else if(tcp->flags() == (TCP::SYN | TCP::ACK)) {
            if(flowstate.find(toflow) != flowstate.end()) {

                std::pair<int,int>& state = flowstate[toflow];

                if(state.first != 1) {
//                    std::cerr << "invalid SYN - connection not valid" << std::endl;
                }
                else {
                    state.first = 2;
                    std::vector<tcp_packet>& v = flowdata[std::make_pair(toflow, state.second)].second;
                    tcp_packet t;
                    t.seq = tcp->seq();
                    t.ack = tcp->ack_seq();
                    t.flags = tcp->flags();
                    t.sec = packet.timestamp().seconds();
                    t.usec = packet.timestamp().microseconds();
                    t.l = 0;
                    if(tcp->inner_pdu() != 0)
                      t.l = ip->tot_len() -  20 - 4*tcp->data_offset();
                    v.push_back(t);
                }
            }
            else {
//                std::cerr << "invalid SYN/ACK" << std::endl;
            }
        }
        else {
            if(flowstate.find(fromflow) != flowstate.end() && flowstate[fromflow].first == 2) {
                int id = flowstate[fromflow].second;
                std::pair<int,int>& state = flowstate[fromflow];
                std::vector<tcp_packet>& v = flowdata[std::make_pair(fromflow, state.second)].first;
                tcp_packet t;
                t.seq = tcp->seq();
                t.ack = tcp->ack_seq();
                t.flags = tcp->flags();
                t.sec = packet.timestamp().seconds();
                t.usec = packet.timestamp().microseconds();
                t.l = 0;
                if(tcp->inner_pdu() != 0)
                  t.l = ip->tot_len() -  20 - 4*tcp->data_offset();
                v.push_back(t);
            }
            else if(flowstate.find(toflow) != flowstate.end() && flowstate[toflow].first == 2) {
                int id = flowstate[toflow].second;
                std::pair<int,int>& state = flowstate[toflow];
                std::vector<tcp_packet>& v = flowdata[std::make_pair(toflow, state.second)].second;
                tcp_packet t;
                t.seq = tcp->seq();
                t.ack = tcp->ack_seq();
                t.flags = tcp->flags();
                t.sec = packet.timestamp().seconds();
                t.usec = packet.timestamp().microseconds();
                t.l = 0;
                if(tcp->inner_pdu() != 0)
                  t.l = ip->tot_len() -  20 - 4*tcp->data_offset();
                v.push_back(t);
            }
        }
    }

    return true;
    });

    std::ofstream f;
    std::cerr << "write into " << output_filename.str() << std::endl;
    f.open(output_filename.str());

    //ugly manual JSON creation - JSON libraries needed enormous amounts of memory and were slow
    f << "[" << std::setprecision(6) << std::fixed << std::endl;

    bool first = true;

    for (auto flow_it = floworder.begin(); flow_it != floworder.end(); ++flow_it) {
        std::pair<flow_ident, int>& p = *flow_it;
        if(first) {
            first = false;
        }
        else {
            f << ",";
        }
        f << "[" << std::endl;
        f << " [" << std::endl;
        f << "  [" << std::endl;

        for(auto p_it = flowdata[p].first.begin(); p_it != flowdata[p].first.end(); ++p_it) {
            tcp_packet& tp = *p_it;
            double t = (tp.sec * 1.0) + (tp.usec * 1.0) * 1e-6;
            f << "   {\"seq\":" << tp.seq << ", \"ack\":" << tp.ack << ", \"flags\":" << tp.flags << ", \"l\":" << tp.l << ", \"t\":" << t << "}";
            if(p_it + 1 != flowdata[p].first.end()) {
                f << ",";
            }
            f << std::endl;
        }

        f << "  ]," << std::endl << "  [" << std::endl;

        for(auto p_it = flowdata[p].second.begin(); p_it != flowdata[p].second.end(); ++p_it) {
            tcp_packet& tp = *p_it;
            double t = (tp.sec * 1.0) + (tp.usec * 1.0) * 1e-6;
            f << "   {\"seq\":" << tp.seq << ", \"ack\":" << tp.ack << ", \"flags\":" << tp.flags << ", \"l\":" << tp.l << ", \"t\":" << t << "}";
            if(p_it + 1 != flowdata[p].second.end()) {
                f << ",";
            }
            f << std::endl;
        }
        f << "  ]" << std::endl << " ]," << std::endl;
        f << " \"" << filename << "\"," << std::endl;
        f << " [" << std::endl;
        f << "  [" << std::endl;
        f << "   \"" << std::get<0>(p.first) << "\"," << std::endl << "   \"" << std::get<1>(p.first) << "\"," << std::endl << "   " << std::get<2>(p.first) << "," << std::endl << "   " << std::get<3>(p.first) << std::endl;
        f << "  ]" << std::endl;
        f << "  ," << std::endl;
        f << "  " << p.second << std::endl;
        f << " ]" << std::endl;
        f << "]" << std::endl;
    }
    f << "]" << std::endl;

}
std::vector<std::string> filename_queue;

void* worker(int t) {
    std::string filename;
    std::map<flow_ident, std::pair<int,int>> flowstate;
    std::map<std::pair<flow_ident, int>, std::pair<std::vector<tcp_packet>,std::vector<tcp_packet>>> flowdata;
    std::vector<std::pair<flow_ident, int>> floworder;
    for(int i = t; i < filename_queue.size(); i += NUM_THREADS) {
        try {
            analyze(filename_queue[i], flowstate, flowdata, floworder);
        }
        catch(...) {
            std::cerr << "analyze failed" << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    //https://stackoverflow.com/questions/8401777/simple-glob-in-c-on-unix-system
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    int return_value = glob("../data/pcaps/*.pcap", GLOB_TILDE, NULL, &glob_result);

    if(return_value != 0) {
        globfree(&glob_result);
        std::cerr << "glob error" << std::endl;
        exit(1);
    }

    std::cerr << glob_result.gl_pathc << " files - start" << std::endl;

    for(size_t i = 0; i < glob_result.gl_pathc; ++i) {
        std::string s(glob_result.gl_pathv[i]);
        filename_queue.push_back(s);
    }

    pid_t pids[NUM_THREADS];

    //tins segfaults when using in threads - therefore start different processes using fork
    for (int t=0; t<NUM_THREADS; t++){
        if (pids[t]=fork()==0) {
            //child
            worker(t);
            exit(0);
        } else {
            //parent
        }
    }

    for(int t = 0; t < NUM_THREADS; t++) {
       waitpid(pids[t],0,0);
    }

    std::cout << "finished" << std::endl;
    globfree(&glob_result);
}
