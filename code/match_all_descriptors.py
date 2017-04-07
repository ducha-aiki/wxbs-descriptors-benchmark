#!/usr/bin/python2 -utt
# -*- coding: utf-8 -*-
from utils import w1bs
DESCS_DIR = "../data/out_descriptors"
REWRITE_EXISTING = False
dist_dict = {"BRIEF": "Hamming"}
force_rewrite_list = []#["BRIEF"]
if __name__ == "__main__":
    w1bs.match_descriptors_and_save_results(DESC_DIR = DESCS_DIR, do_rewrite = REWRITE_EXISTING, dist_dict = dist_dict, force_rewrite_list = force_rewrite_list)
