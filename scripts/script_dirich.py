import os
import numpy as np


results = "output_collas"
datasets = ["imagenet_c", "cifar100_c", "cifar100_ccorrel_10.0", "cifar100_ccorrel_1.0",
            "cifar100_ccorrel_0.1", "cifar100_ccorrel_0.01"]
methods = ["memo"]

print_summary = True
add_cross_val_metric = True

seeds = [x+1 for x in range(3)]

measures = ["SND", "CONSISTENCY", "ENTROPY", "SOURCE_mean", "SMALL_mean_error", "mean error"]

cross_datasets = {"cifar100_c" : "cifar100_c", #THIS IS ORACLE!
                  "cifar100_ccorrel_10.0" : "imagenet_c",
                  "cifar100_ccorrel_1.0" : "imagenet_c",
                  "cifar100_ccorrel_0.1" : "imagenet_c",
                  "cifar100_ccorrel_0.01" : "imagenet_c",
                  "imagenet_c" : "cifar100_c",
                  "imagenet_c_x10" : "imagenet_c",
                  "domainnet126" : "imagenet_c",
                  "clad": "imagenet_c",
                  "imagenet_r" : "imagenet_r"}

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
                    lines = open(os.path.join(root, file_to_read)).readlines()
                    if len(lines) > 7:
                        if not "wall" in lines[-7] or len(lines) < 7:
                            print("skipping ",root,  file_to_read)
                            #raise("x")
                            continue
                    else:
                        continue

                    #print(lines[-6]) #TODO, add to metrics
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
for dataset, method_results in final_results.items():
    #print("method ", method)
    for method, seed_results in method_results.items():
        for seed, hparam_results in seed_results.items():   
            for parameter, metrics_results in hparam_results.items():
                mean_error = np.mean(metrics_results["mean error"])
                acc = 100. - mean_error
                for metric in measures: 
                    value = np.mean(metrics_results[metric]) #WE IGNORE THIS ONE CURRENTLY, BUT WE COULD PLOT VALUE VS ACC
                    if not metric in res_measures[dataset][method][seed]:
                        res_measures[dataset][method][seed][metric] = ('None', 999999)
                    _, curr_value = res_measures[dataset][method][seed][metric]
                    #print(parameter, "current ", curr_acc, "new ", acc, metric)
                    if value < curr_value:
                        #print("here")
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
            
avg_measures = {}

all_measures = measures + ["median"]
if add_cross_val_metric:
    all_measures = all_measures + ["cross_val"]
er = 0
for dataset, methods in res_measures.items():
    for method, method_res in methods.items():
        per_metric = {}
        print(dataset, method)
        if not method in avg_measures:
            avg_measures[method] = {}
        for seed, seed_res in method_res.items():
            for metric, tuples in seed_res.items():
                #print(metric, tuples)
                if 'median' in tuples[0]:
                    mean_error = tuples[1]
                else:
                    #for given metric, best parameters are in tuples[0]. Tuples[1] is measure value
                    try:
                        mean_error = final_results[dataset][method][seed][tuples[0]]["mean error"]
                    except:
                        er += 1
                        print(dataset, method, seed, tuples[0], "not available")
                        #print("available keys are ", len(final_results[dataset][method][seed]), final_results[dataset][method][seed].keys())
                        mean_error = 100.
                acc = 100. - mean_error
              #  print(metric, "res ", acc)
                if not metric in per_metric:
                    per_metric[metric] = [acc]
                else:
                    per_metric[metric].append(acc)
           
        for metric in all_measures:
            curr = per_metric[metric]
            print(metric, np.mean(curr), np.std(curr), curr)
            if not "cifar" in dataset: #HARDCODED TO BE CIFAR CORRELATED
                continue
            if not metric in avg_measures[method]:
                avg_measures[method][metric] = {}
            assert(dataset not in avg_measures[method][metric])
            avg_measures[method][metric][dataset] = np.mean(curr)

print(er)

print("\n Plotting results \n")
for measure in all_measures:
    lll = []
    for dataset in datasets[1:]:
        res = avg_measures[method][measure][dataset]
        lll.append(res)
    print(measure, lll)
