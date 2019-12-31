# DeePCCI framework

Code for our [DeePCCI paper](https://dl.acm.org/citation.cfm?doid=3341216.3342211)

**Note: This repo was rebased to remove the git lfs files due to the github git lfs quota**

**Weight and dataset files available at:** https://git-ce.rwth-aachen.de/constantin.sander/deepcci-public

## Quick Overview
* **generate**: generation of pcaps
  * create params (which network topologies and params to use for emulation)  using `python3 create_params.py`
  * compile our small trafficgenerator using `make`
  * create pcaps with mininet according to params using `python2 create.py` (ouputs to `data/pcaps`)
* **preprocess**: parsing pcaps to arrival jsons
  * make preprocessor using `make`
  * run `./preprocess` which will process all pcaps under `data/pcaps` ( outputs to `data/preprocessed`)
* **process**: convert preprocessed json to histogram dataset
  * run `python3 create_features.py` to create histogram files of preprocessed pcaps ( outputs to `data/processed`)
    * the script outputs a histogram csv file for every pcap and adds labelfiles for our test, train and validation data
* **learn**: deep learning model (tensorflow v1)
  * `dataset.zip` and `weights.zip` contain our dataset and our weights used for the paper
  * run `python3 train.py --config configs/deepcci.json --train` for training
    * it will save the weights for every epoch and the weights of the best loss and accuracy achieved
  * run `python3 train.py --config configs/deepcci.json --test --model ../data/weights/deepcci` for testing
    * it will create a json file which contains all logits for every dataset entry, which can be used for later evaluation

## Dependencies
* `pip3 install tensorflow-gpu==1.14 numpy sklearn`
* mininet - please clone and build from https://github.com/mininet/mininet
* `pip2 install future` (python2 dependency due to mininet)
* `sudo apt-get install libtins-dev`

## Details
### generate
Mininet based network emulation

We emulate different networks according to our paper and record traffic before and after the according bottleneck links into pcaps.
Mininet does not support python3, so python2 has to be used with the future lib to support certain python3 semantics. We recommend to compile mininet from source as we got multiple errors from mininet functions when using the ubuntu package.

The `create_params.py` script will create multiple json files, which contain topology information for every pcap.
The `create.py` script will use theses json files to generate pcaps accordingly. To generate the traffic, we use a small C program which simply listens for incoming connections in server mode and sends data as fast as possible in client mode (comparable to iperf). Our traffic generator can select the congestion control used and allows to enable pacing per socket option. However, we found that the socket option under Linux 4.18 behaves unexpectedly and we use tc-fq on the sender link instead.

### preprocess
We preprocess the recorded pcaps into json files which we found to be faster parseable with python than pcaps with scapy.

For this, we use a C++ program with libtins to parse the pcaps and extract its flows.

### process
Python based processing of preprocessed traffic. We generate our histograms here + labelfiles for the training.

Every pcap file gets its own histogram csv file. Linking the different files for training, validation and testing, we use labelfiles which contain the used congestion control parameters per line and link to the histogram csv files.

### learn
We use tensorflow v1 to train our deep neural network and for inference.

We used tensorflow with GPU support on Nvidia GPUs and use the optimized CuDNN implementation of LSTMs.
As we do not truncate our time series data and use batch sizes of 32, we require rather large amounts of GPU memory during training.
We used Nvidia Tesla V100 GPUs with 16GB of VRAM.

A `machine.settings` file can be created, to switch to the tensorflow implementation of LSTMs for inference on CPUs (not tested for learning). 
