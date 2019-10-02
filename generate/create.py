import sys

sys.path.append("..")

from generate.json_parametersource import JsonParams
from generate.mininet_emulation import do, cleanup

datadir = "../data/pcaps/"
jsondir = "../data/jsons/"
tmpdir = "/tmp/"


def run(params):
    cleanup()
    do(datadir, jsondir, tmpdir, params)


def main():
    ps = JsonParams(run)
    ps.read_and_start()
    print("finished")


if __name__ == "__main__":
    main()
