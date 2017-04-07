#!/usr/bin/python2 -utt
# -*- coding: utf-8 -*-
from utils import w1bs
DESCS_DIR = "../data/out_descriptors"
GRAPHS_DIR = "../data/out_py_graphs"

descs_to_draw = ["TFeat","SIFT", "Pixels11", "BRIEF"]

if __name__ == "__main__":
    methods = ["SNN_ratio"]#, "Distance", "SNN_distance_difference"]
    w1bs.draw_and_save_plots(DESCS_DIR,  OUT_DIR = GRAPHS_DIR, methods = methods, descs_to_draw = descs_to_draw)
    
