#!/bin/bash
./describe_patches_by_all_descriptors.py
./match_all_descriptors.py
./draw_all_plots.py
cd ../data/out_py_graphs/; for f in *.eps ; do epstopdf $f ;done; rm *.eps; cd -
