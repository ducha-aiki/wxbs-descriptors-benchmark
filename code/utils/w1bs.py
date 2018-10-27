import sys
import fnmatch
import os
import numpy as np
from subprocess import call
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances
import math
import time
import colorsys
import matplotlib.pyplot as plt
def get_list_of_patch_images(DATASET_DIR = "../data/W1BS",  mask = '*.bmp'):
    patch_images = []
    for root, dirnames, filenames in os.walk(DATASET_DIR):
        for filename in fnmatch.filter(filenames, mask):
            patch_images.append(os.path.join(root, filename))
    return patch_images
def get_list_of_descriptor_scripts(SCRIPTS_DIR):
    onlyfiles = [os.path.join(SCRIPTS_DIR, f) for f in os.listdir(SCRIPTS_DIR) if (os.path.isfile(os.path.join(SCRIPTS_DIR, f)) and f != "__init__.py")]   
    return onlyfiles
def getDescExtension(desc_fname):
    desc_basename = os.path.basename(desc_fname)
    desc_name = os.path.splitext(desc_basename)[0]
    return desc_name
def describe_patch_images_with_descriptor(desc_fname, img_files_list, OVERWRITE_IF_EXISTS = True, list_of_descs_overwrite_anyway = []):
    desc_name = getDescExtension(desc_fname)
    for img_fname in img_files_list:
        out_fname = img_fname.replace("data/W1BS", "data/out_descriptors").replace(".bmp", "." + desc_name)
        process = OVERWRITE_IF_EXISTS or (not os.path.isfile(out_fname)) or (desc_name in  list_of_descs_overwrite_anyway)
        out_dir = os.path.dirname(out_fname)
        if len(out_dir) > 0:
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
        if process:
            command_list = ["./"+desc_fname, img_fname, out_fname]
            print (command_list)
            out = call(command_list)
            print (out)
    return
def checkIfDescriptorScriptIsOK(desc_fname):
    print ("Checking", desc_fname)
    desc_name = getDescExtension(desc_fname)
    status = True
    try:
        describe_patch_images_with_descriptor(desc_fname, ["graf_test.bmp"])
        with open("graf_test." + desc_name, "rb") as f:
            lines = f.readlines()
        if len(lines) != 2:
            print ("Number of descriptors", len(lines), "!= number of patches 2")
            status =  False
        descs = []
        for l in lines:
            d1 = np.fromstring(l, dtype=float, sep=' ')
            descs.append(d1)
        if len(descs[0]) != len(descs[1]):
            print ("Descriptors have different lengths", descs[0].shape, descs[1].shape)
            status = False
    except:#Exception, e:
        #print (str(e))
        status =  False
    if status:
        os.remove("graf_test." + desc_name)
    return status
def match_descriptors(d1, d2, metric = "L2", batch_size = 256):
    out_dict = {}
    eps = 1e-12;
    d1_num, d1_dim = d1.shape
    d2_num, d2_dim = d2.shape
    if d1_dim != d2_dim:
        print ("d1 and d2 have different dimentions", d1_dim, d2_dim)
        return None
    out = np.zeros((d1_num,7)) # idx_in_d1 indx_in_d2 dist dist_ratio 2nd_idx_in_d2 dist2  dist_diff
    if metric == "L2":
        d1_norm_squared = (d1**2).sum(axis=1)
        d2_norm_squared = (d2**2).sum(axis=1)
    n_batches = int(math.floor(float(d1_num) / float(batch_size)))
    t=time.time()
    for batch_idx in range(n_batches):
        curr_idxs = np.arange(batch_idx*batch_size,(batch_idx+1)*batch_size);
        query = d1[curr_idxs,:]
        if metric == "L2":
            dists =  euclidean_distances(query,d2, squared = False, 
                                         X_norm_squared = d1_norm_squared[curr_idxs].reshape(-1,1),
                                         Y_norm_squared = d2_norm_squared)
        elif metric == "Hamming":
            dists = pairwise_distances(query,d2,metric = "hamming")
        else:
            print ("Non implemented yet")
            return None
        out[curr_idxs,0] = curr_idxs;
        out[curr_idxs,1] = np.argmin(dists, axis = 1)
        ravel_idx = np.ravel_multi_index([np.arange(batch_size), 
                                                         out[curr_idxs,1].astype(np.int32)], (batch_size,d2_num))
        out[curr_idxs,2] = dists.ravel()[ravel_idx]
        dists.ravel()[ravel_idx]  = 1e7;
        out[curr_idxs,4] = np.argmin(dists, axis = 1)
        ravel_idx2 = np.ravel_multi_index([np.arange(batch_size), 
                                                         out[curr_idxs,4].astype(np.int32)], (batch_size,d2_num))
        out[curr_idxs,5] = dists.ravel()[ravel_idx2]
        out[curr_idxs,3] = out[curr_idxs,2] / (out[curr_idxs,5] + eps)
    last_batch_idxs = np.arange(n_batches*batch_size,d1_num);
    n_last = len(last_batch_idxs)
    if n_last > 0:
        query = d1[last_batch_idxs,:]#.reshape(-1,self.dim)
        if metric == "L2":
            dists =  euclidean_distances(query,d2, squared = False, 
                                            X_norm_squared = d1_norm_squared[last_batch_idxs].reshape(-1,1),
                                            Y_norm_squared = d2_norm_squared)
        elif metric == "Hamming":
            dists = pairwise_distances(query,d2,metric = "hamming")
        else:
            print ("Non implemented yet")
            return None
        out[last_batch_idxs,0] = last_batch_idxs;
        out[last_batch_idxs,1] = np.argmin(dists, axis = 1) 
        ravel_idx = np.ravel_multi_index([np.arange(n_last), 
                                                             out[last_batch_idxs,1].astype(np.int32)], (n_last,d2_num))
        out[last_batch_idxs,2]  =  dists.ravel()[ravel_idx]

        dists.ravel()[ravel_idx]  = 1e7;
        out[last_batch_idxs,4] = np.argmin(dists, axis = 1)
        ravel_idx2 = np.ravel_multi_index([np.arange(n_last), 
                                                             out[last_batch_idxs,4].astype(np.int32)], (n_last,d2_num))
        out[last_batch_idxs,5]  =  dists.ravel()[ravel_idx2]
        out[last_batch_idxs,3] = out[last_batch_idxs,2] / (out[last_batch_idxs,5] + eps)
    out_dict["idx1"] = out[:,0].astype(np.int32)
    out_dict["nearest_idx2"] = out[:,1].astype(np.int32)
    out_dict["nearest_dist"] = out[:,2]
    out_dict["2ndnearest_ratio"] = out[:,3]
    out_dict["2ndnearest_idx2"] = out[:,4].astype(np.int32)
    out_dict["2ndnearest_dist"] = out[:,5]
    out[:,6] =  out[:,5] - out[:,2]
    out_dict["2ndnearest_diff"] = out[:,6]
    return out_dict,out
def get_list_of_pairing_files_with_descriptors(DESC_DIR = "../data/out_descriptors"):
    desc_pair_files = []
    for root, dirnames, filenames in os.walk(DESC_DIR):
        for fname1 in filenames:
            fname1 = os.path.join(root, fname1)
            if "/1/" in fname1:
                fname2 = fname1.replace("/1/", "/2/")
                if os.path.isfile(fname2):
                    desc_pair_files.append((fname1,fname2))
                else:
                    print ("Warning! No pair for", fname1)
    return desc_pair_files
def match_descriptors_and_save_results(DESC_DIR = "../data/out_descriptors", do_rewrite = False, dist_dict = {}, force_rewrite_list = []):
    desc_pair_files = get_list_of_pairing_files_with_descriptors(DESC_DIR = DESC_DIR)
    for pair in desc_pair_files:
        match_fname = pair[0].replace("/1/", "/match/") + ".match"
        our_dir = os.path.dirname(match_fname)
        if not os.path.isdir(our_dir):
            os.makedirs(our_dir)
        print (match_fname)
        desc_name = pair[0].split(".")[-1];
        needs_matching = (desc_name in force_rewrite_list) or (not os.path.isfile(match_fname)) or do_rewrite
        if needs_matching:
            #if match_fname.endswith("TFeatREF.match"):
            #    d1 = np.nan_to_num(np.loadtxt(pair[0], delimiter = ",").astype(np.float64))
            #    d2 = np.nan_to_num(np.loadtxt(pair[1], delimiter = ",").astype(np.float64))
            #elif match_fname.endswith("SIFTREF.match"):
            #    d1 = np.nan_to_num(np.loadtxt(pair[0], delimiter = ";").astype(np.float64))
            #    d2 = np.nan_to_num(np.loadtxt(pair[1], delimiter = ";").astype(np.float64))
            #else:
            d1 = np.nan_to_num(np.loadtxt(pair[0]).astype(np.float64))
            d2 = np.nan_to_num(np.loadtxt(pair[1]).astype(np.float64))
            t = time.time()
            if desc_name in dist_dict:
                metric = dist_dict[desc_name]
            else:
                metric = "L2"
            match_dict, match_matrix = match_descriptors(d1, d2, metric = metric)
            el = time.time() - t
            print ( el, "sec")
            np.savetxt(match_fname, match_matrix, delimiter=' ', fmt='%10.5f')
    return
def get_recall_and_pecision(dist, is_correct, n_pts = 100, smaller_is_better = True):
    recall = np.zeros((n_pts))
    precision = np.zeros((n_pts))
    max_correct = float(len(dist))
    if smaller_is_better:
        thresholds =  np.linspace(0,dist.max(),n_pts)
        for i in range(len(thresholds)):
            recall[i] = np.sum(is_correct * (dist < thresholds[i]).astype(np.float32)) / max_correct;
            den = np.sum(dist < thresholds[i]).astype(np.float32)
            #print recall[i], thresholds[i],den
            if den > 0:
                precision[i] =  np.sum(is_correct * (dist < thresholds[i]).astype(np.float32)) / den;
            else:
                precision[i] = 1.
    else:
        thresholds =  np.linspace(dist.max(),0,n_pts)
        for i in range(len(thresholds)):
            recall[i] = np.sum(is_correct * (dist > thresholds[i]).astype(np.float32)) / max_correct;
            den = np.sum(dist > thresholds[i]).astype(np.float32)
            #print recall[i], thresholds[i],den
            if den > 0:
                precision[i] =  np.sum(is_correct * (dist > thresholds[i]).astype(np.float32)) / den;
            else:
                precision[i] = 1.
    ap = np.trapz (recall, x = 1 - precision)
    #print ap
    return recall,precision,ap
def get_rec_prec_of_all_match_files(DESC_DIR = "../data/out_descriptors"):
    match_files_per_desc_per_set_per_pair = {}
    for root, dirnames, filenames in os.walk(DESC_DIR):
        for fname1 in filenames:
            if ("/match" in root) and fname1.endswith(".match"):
                dataset = root.split('/')[-2]
                pair = fname1.split('.')[0]
                descriptor = fname1.split('.')[-2]
                #print dataset,pair,descriptor
                if descriptor not in match_files_per_desc_per_set_per_pair:
                    match_files_per_desc_per_set_per_pair[descriptor] = {}
                desc_f = match_files_per_desc_per_set_per_pair[descriptor]
                if dataset not in desc_f:
                    desc_f[dataset] = {}
                match_files_per_desc_per_set_per_pair[descriptor][dataset][pair] = os.path.join(root,fname1)
    return match_files_per_desc_per_set_per_pair
def get_rec_prec_ap_for_all_match_files(DESC_DIR, whitelist = []):
    files_dict = get_rec_prec_of_all_match_files(DESC_DIR = DESC_DIR)
    results_dict = {}
    for desc_name, v in files_dict.items():
        if (len(whitelist) > 0) and desc_name in whitelist:
            results_dict[desc_name] = {}
            for dataset_name, vv in v.items():
                results_dict[desc_name][dataset_name] = {}
                for pair_name, fname in vv.items():
                    results_dict[desc_name][dataset_name][pair_name]= {}
                    m = np.loadtxt(fname)
                    is_correct = (m[:,0] == m[:,1]).astype(np.float32)
                    r,p,ap = get_recall_and_pecision(m[:,3],is_correct)
                    results_dict[desc_name][dataset_name][pair_name]["SNN_ratio"] = (r,p,ap)
                    r,p,ap = get_recall_and_pecision(m[:,2],is_correct)
                    results_dict[desc_name][dataset_name][pair_name]["Distance"] = (r,p,ap)
                    r,p,ap = get_recall_and_pecision(m[:,6],is_correct, smaller_is_better = False)
                    results_dict[desc_name][dataset_name][pair_name]["SNN_distance_difference"] = (r,p,ap)
    return results_dict
def get_average_plot_data_for_datasets(full_results_dict, method = "SNN_ratio"):
    avg_res = {}
    avg_res["Total"]  = {}
    for desc_name, v in full_results_dict.items():
        total_start = True
        total_r = 0
        total_p = 0
        total_ap = 0
        total_count = 0
        #print (desc_name)
        for dataset_name, vv in v.items():
            #print dataset_name
            if dataset_name not in avg_res:
                avg_res[dataset_name] = {}
            num_pairs = float(len(vv))
            r = 0
            p = 0
            ap = 0
            for pair_name, vvv in vv.items():
                r += vvv[method][0]
                p += vvv[method][1]
                ap += vvv[method][2]
            r /= num_pairs
            p /= num_pairs
            ap /= num_pairs
            total_r += r
            total_p += p
            total_ap += ap
            total_count += 1
            avg_res[dataset_name][desc_name] = (r,p,ap)
        total_r /= float(total_count)
        total_p /= float(total_count)
        total_ap /= float(total_count)
        avg_res["Total"][desc_name] = (total_r, total_p, total_ap)
    return avg_res
def draw_and_save_plots(DESC_DIR, OUT_DIR = "../data/out_graphs", methods = ["SNN_ratio"], colors = [], lines = [], descs_to_draw = []):
    full_results_dict = get_rec_prec_ap_for_all_match_files(DESC_DIR, whitelist = descs_to_draw)
    #print ('tttt',full_results_dict)
    desc_to_color = {}
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    hue = 0
    for m in methods:
        avg_res = get_average_plot_data_for_datasets(full_results_dict, method = m)
        #print (avg_res)
        for dataset_name, v in avg_res.items():
            out_fname = dataset_name + "_" + m + '.eps'
            out_path = os.path.join(OUT_DIR,out_fname)
            plt.figure(dpi=300,  figsize=(8, 6))
            idx = 0
            leg = []
            if len(descs_to_draw) > 0:
                colors_num = len(descs_to_draw);
            else:
                colors_num = len(v);
            r = 0
            p = 0
            ap = 0
            for desc_name, vv in v.items():
                if len(descs_to_draw)>0:
                    if desc_name not in descs_to_draw:
                        continue
                try:
                    ttttt = desc_to_color[desc_name]
                except:
                    color = list ( colorsys.hsv_to_rgb(hue,1.0,1.0)   )
                    desc_to_color[desc_name] = color
                    hue += 1.0/float(colors_num)
                    print( "adding", desc_name)
                r,p,ap = vv;
                print (desc_name, dataset_name, "%.5f" % ap)
                pl = plt.plot(1. - p, r, color = desc_to_color[desc_name])
                leg.append(desc_name + " , mAUC = " + "%.3f" % ap)
            plt.xlabel("1 - precision")
            plt.ylabel("Recall")
            plt.legend(leg,prop={'size':8},loc = 'best')
            #print (leg)
            plt.savefig(out_path)
            plt.clf()
    return   
def draw_and_save_plots_with_loggers(DESC_DIR, OUT_DIR = "../data/out_graphs", methods = ["SNN_ratio"],
                        colors = [], lines = [], descs_to_draw = [],really_draw = False,
                        logger=None,
                        tensor_logger = None):
    full_results_dict = get_rec_prec_ap_for_all_match_files(DESC_DIR, whitelist = descs_to_draw)
    desc_to_color = {}
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    hue = 0
    for m in methods:
        avg_res = get_average_plot_data_for_datasets(full_results_dict, method = m)
        for dataset_name, v in avg_res.items():
            out_fname = dataset_name + "_" + m + '.eps'
            out_path = os.path.join(OUT_DIR,out_fname)
            if really_draw:
                plt.figure(dpi=300,  figsize=(8, 6))
            idx = 0
            leg = []
            if len(descs_to_draw) > 0:
                colors_num = len(descs_to_draw);
            else:
                colors_num = len(v);
            r = 0
            p = 0
            ap = 0
            for desc_name, vv in v.items():
                if len(descs_to_draw)>0:
                    if desc_name not in descs_to_draw:
                        continue
                try:
                    ttttt = desc_to_color[desc_name.replace("Inv", "").replace("Mirr","")]
                except:
                    color = list ( colorsys.hsv_to_rgb(hue,1.0,1.0)   )
                    desc_to_color[desc_name.replace("Inv", "").replace("Mirr","")] = color
                    hue += 1.0/float(colors_num)
                    print ("adding", desc_name)
                r,p,ap = vv;
                print (desc_name, dataset_name, p[0])
                if ('Inv' in desc_name) or ('Mirr' in desc_name):
                    if really_draw:
                        pl = plt.plot(1. - p, r, color = desc_to_color[desc_name.replace("Inv", "").replace("Mirr","")], linestyle = "--")
                else:
                    if really_draw:
                        pl = plt.plot(1. - p, r, color = desc_to_color[desc_name])
                leg.append(desc_name + " , mAUC = " + "%.4f" % ap)
                if (tensor_logger != None):
                    tensor_logger.log_value(desc_name+' '+dataset_name, ap).step()
                if (logger != None):
                        logger.log_stats('/logs', ' Matching stats: ', str(desc_name +' '+dataset_name+" , mAUC = " + "%.4f" % ap))

            print(leg)
            if really_draw:
                plt.xlabel("1 - precision")
                plt.ylabel("Recall")
                plt.legend(leg,prop={'size':8},loc = 'best')
                plt.savefig(out_path)
                plt.clf()
    return 
