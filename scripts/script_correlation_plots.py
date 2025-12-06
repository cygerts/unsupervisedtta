import os
import numpy as np


results = "output_collas"
datasets = ["cifar100_c", "imagenet_c", "domainnet126", "imagenet_r", "imagenet_v2"]
methods = ["adacontrast", "eata", "sar", "rmt_SF", "tent", "memo"]

print_summary = True
add_cross_val_metric = False

seeds = [x+1 for x in range(3)]

measures = ["SND", "CONSISTENCY", "ENTROPY", "SOURCE_mean", "SMALL_mean_error", "mean error"]

cross_datasets = {"cifar100_c" : "imagenet_c",
                  "imagenet_c" : "cifar100_c",
                  "imagenet_c_x10" : "imagenet_c",
                  "cifar100_c_x10" : "cifar100_c",
                  "cifar100_c_bs32" : "imagenet_c",
                  "cifar100_c_bs64" : "imagenet_c",
                  "cifar100_c_bs128" : "imagenet_c",
                  "imagenet_r_bs32" : "imagenet_c",
                  "imagenet_r_bs64" : "imagenet_c",
                  "imagenet_r_bs128" : "imagenet_c",
                  "imagenet_a" : "imagenet_c",
                  "domainnet126" : "imagenet_c",
                  "clad": "imagenet_c",
                  "imagenet_r" : "imagenet_c",
                  "imagenet_r_x10" : "imagenet_r",
                  "imagenet_v2" : "imagenet_c"}

final_results = {}  #dataset x method x seed x params_key x metrics
h_params_keys = {} #method x seed x hparamskey #helper dict. THIS TELLS for each method (and seed), what are available hparams paths. SHould be the same across seeds!

for dataset in datasets:
    dataset_dir = os.path.join(results, dataset)
    if not dataset in final_results:
        final_results[dataset] = {}
    for seed in seeds:
        seed_dir = os.path.join(dataset_dir, "seed_" + str(seed))
        for method in methods:
            curr_dir = os.path.join(seed_dir, method)
            #print("parsing ", curr_dir)
            for root, dirs, files in os.walk(curr_dir):
                if len(files) > 0:
                    if not method in final_results[dataset]:
                        final_results[dataset][method] = {}
                    if not seed in final_results[dataset][method]:
                        final_results[dataset][method][seed] = {}
                    if not method in h_params_keys:
                        h_params_keys[method] = {}
                    if not seed in h_params_keys[method]:
                        h_params_keys[method][seed] = []
                    files.sort()
                    file_to_read = files[-1] #THE NEWEST FILE
                   # print(root, files)
                    method_idx = root.rfind(method) + len(method) + 1
                    params_key = root[method_idx:]
                    #print(root, file_to_read)
                    lines = open(os.path.join(root, file_to_read)).readlines()
                    if len(lines) > 7:
                        if not "wall" in lines[-7] or len(lines) < 7:
                            print("skipping ",root,  file_to_read)
                            continue
                    else:
                        continue

                    for idx, metric in enumerate(measures):

                        if not params_key in final_results[dataset][method][seed]:
                            final_results[dataset][method][seed][params_key] = {}
                        if not params_key in h_params_keys[method]:
                            h_params_keys[method][seed].append(params_key)
                        try:
                            assert(metric in lines[-idx -1])
                        except:

                            print(method, "err ", params_key, metric, lines[-idx -1])
                            continue
                        lines[-idx -1] = lines[-idx -1].replace("%%","%") #SOME LOGGING ERROR, FIXING HERE
                        idx1 = lines[-idx -1].rfind(" ")
                        idx2 = lines[-idx -1].rfind("%")
                        value = float(lines[-idx -1][idx1+1:idx2])
                        final_results[dataset][method][seed][params_key][metric] = value
                    #print("done ", file_to_read)
res_measures={} #dataset x method x seed x metrix x (hparam, acc)


#final_results = {}  #dataset x method x seed x hparams_key x metrics
for dataset, method_results in final_results.items():
    if not dataset in res_measures:
        res_measures[dataset] = {}
    for method, seed_results in method_results.items():
        if not method in res_measures[dataset]:
            res_measures[dataset][method] = {}
        for seed, hparam_results in seed_results.items():
            if not seed in res_measures[dataset][method]:
                res_measures[dataset][method][seed] = {}
            mean_accs = []
            for hparam, metric_results in hparam_results.items():
                #print(metric_results)
                mean_error = metric_results["mean error"]
                mean_accs.append(100. - mean_error)
                #print(param_results)      
            if print_summary and len(mean_accs) > 0:
                print(dataset, method, seed, "LEN = ", len(mean_accs), np.max(mean_accs), mean_accs)
            else:
                print(dataset, method, seed, "Len = 0s")
            res_measures[dataset][method][seed]["median"] = ('median_{}'.format(len(mean_accs)), 100. - np.median(mean_accs))
            
print('\n')

xxx = {} #dataset x method x metric x metric-val
yyy = {} #dataset x method x metric x acc

for dataset, method_results in final_results.items():
    if not dataset in xxx:
        xxx[dataset] = {}
        yyy[dataset] = {}
        
    #print("method ", method)
    for method, seed_results in method_results.items():
        if not method in xxx[dataset]:
            xxx[dataset][method] = {}
            yyy[dataset][method] = {}
            
        for seed, hparam_results in seed_results.items():   
            for parameter, metrics_results in hparam_results.items():
                mean_error = np.mean(metrics_results["mean error"])
                acc = 100. - mean_error
                for metric in measures: 
                    if not metric in xxx[dataset][method]:
                        xxx[dataset][method][metric] = []
                        yyy[dataset][method][metric] = []
                    value = np.mean(metrics_results[metric]) 
                    if metric in ["SOURCE_mean", "SMALL_mean_error", "mean error"]:
                        value = 100. - value
                    else:
                        value = -value / 100.
                    xxx[dataset][method][metric].append(value)
                    yyy[dataset][method][metric].append(acc)
                    if not metric in res_measures[dataset][method][seed]:
                        res_measures[dataset][method][seed][metric] = ('None', 999999)
                    _, curr_value = res_measures[dataset][method][seed][metric]
                    #print(parameter, "current ", curr_acc, "new ", acc, metric)
                    if value < curr_value:
                     
                        res_measures[dataset][method][seed][metric] = (parameter, value) 

if add_cross_val_metric:
    for dataset, method_results in final_results.items():
        cross_dataset = cross_datasets[dataset]
        for method, seed_results in method_results.items():
            for seed, hparam_results in seed_results.items():   
                #print(res_measures[dataset][method][seed].keys())
                try:
                    cross_hp = res_measures[cross_dataset][method][seed]['mean error']
                    res_measures[dataset][method][seed]["cross_val"] = cross_hp
                except:
                    print(res_measures[cross_dataset][method][seed].keys())

                    print("ERR, crossval won't work ", dataset, cross_dataset, method, seed)
                    continue
            
#res_measures  = dataset x method x seed x metrix x (hparam, acc)
#final_results = dataset x method x seed x hparams_key x metrics

avg_measures = {}
all_measures = measures
if add_cross_val_metric:
    all_measures = all_measures + ["cross_val"]
all_measures = all_measures + ["median"]

er = 0

import matplotlib.pyplot as plt
y_label = {"CONSISTENCY" : "-consistency loss", 
           "ENTROPY" : "- entropy loss", 
           "SOURCE_mean" : "Source accuracy [%]", 
           "SND" : "SND", 
           "SMALL_mean_error" : "100-RND",
           "cross_val" : "C-ACC [%]"}


y_lim = { "CONSISTENCY" : (-2.5, 0.), 
           "ENTROPY" : (-4.5, 0.), 
           "SND" : (0., 3.0),
           "SOURCE_mean" : (0., 100.), 
           "SMALL_mean_error" : (0., 100.),
           "cross_val" : (0., 100.)
        }

font = 14

mdict = {"adacontrast" : "AdaContrast", 
         "eata" : "EATA", 
         "sar" : "SAR", 
         "rmt_SF" : "RMT_SF", 
         "tent" : "TENT", 
         "memo" : "MEMO"}

for measure in all_measures: #except for median and oracle
    if measure in ["median", "mean error"]:
        continue
    for method in methods:
        
        xval = []
        yval = []
        for dataset in datasets:
            xval = xxx[dataset][method][measure]
            yval = yyy[dataset][method][measure]

            plt.scatter(xval, yval, label=dataset)


        plt.ylabel('Accuracy [%]',fontsize=font)
        yl = y_lim[measure]
        plt.xlim(yl[0],yl[1])
        plt.ylim(0,70)
        
        plt.xlabel(y_label[measure],fontsize=font)
        #plt.legend(loc = "upper left", fontsize=font)
        plt.legend(loc = "upper left", fontsize=font)
        plt.title(mdict[method],fontsize=font)
        plt.grid(True)

        # Display the plot or save it to a file
        plt.savefig("plots/{}_{}.png".format(method, measure),dpi=300)
        plt.close()
        print("done")
