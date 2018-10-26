#!/usr/bin/python2 -utt
# -*- coding: utf-8 -*-
from utils import w1bs
DATASET_DIR = "../data/W1BS"
SCRIPTS_DIR  = "descriptors"
OVERWRITE_IF_EXISTS = False
list_of_descs_overwrite_anyway = []
if __name__ == "__main__":
    patch_imgs_fnames = w1bs.get_list_of_patch_images(DATASET_DIR, mask = "*.bmp")
    extractors = w1bs.get_list_of_descriptor_scripts(SCRIPTS_DIR);
    print ("The following descriptors are in ", SCRIPTS_DIR)
    print (extractors)
    valid_extractors = []
    print "Testing..."
    for e in extractors:
        if w1bs.checkIfDescriptorScriptIsOK(e):
            print ("good")
            valid_extractors.append(e)
        else:
            print ("bad")
    for e in valid_extractors:
        print (e, ' is extracting')
        w1bs.describe_patch_images_with_descriptor(e,patch_imgs_fnames, OVERWRITE_IF_EXISTS = OVERWRITE_IF_EXISTS, list_of_descs_overwrite_anyway = list_of_descs_overwrite_anyway)
    
