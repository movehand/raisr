import argparse

def gettrainargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plot", help="Plot the learned filters", action="store_true")
    args = parser.parse_args()
    return args
