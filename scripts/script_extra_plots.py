import os
import numpy as np
from seaborn import scatterplot 
import matplotlib.pyplot as plt
import pandas as pd
#sns.set(style="whitegrid")  # Set the style of the plot

results = "output"
datasets = ["cifar100_c", "imagenet_c", "domainnet126", "imagenet_r"]

print_summary = True
add_cross_val_metric = True

seeds = [x+1 for x in range(3)]

measures = ["CONSISTENCY", "ENTROPY", "SOURCE_mean", "SMALL_mean_error", "mean error"]

cross_datasets = {"cifar100_c" : "imagenet_c",
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
                    #print(root, file_to_read)
                    lines = open(os.path.join(root, file_to_read)).readlines()
                    if not "wall" in lines[-6] or len(lines) < 7:
                        print("skipping ",root,  file_to_read)
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

#FIG. RESULTS BY LR 
data = []
num_datasets = len(datasets)
accs = {} #dataset x method x LR x Acc. All methods aggregated

#final_results = dataset x method x seed x hparams_key x metrics
for dataset, methods in final_results.items():
    print("Dataset ", dataset)
    accs[dataset] = {}
    for method, method_res in methods.items():
        #if not method in accs[dataset]:
        #    accs[dataset][method] = {}
        for seed, seed_res in method_res.items():
            for hparam, hparam_res in seed_res.items():
                for metric, res in hparam_res.items():
                    if metric != "mean error":
                        continue
                    idx = hparam.find("LR") #LR_0.00006250
                    lr = hparam[idx+3:idx+13]
                    mean_error = res
                    #assert(len(mean_error) == 1)
                    acc = 100. - mean_error
                    if method in accs[dataset]:
                        accs[dataset][method].append((lr,acc))
                    else:
                        accs[dataset][method] = [(lr,acc)]

accs2 = {}
for dataset, methods in res_measures.items():
    print("Dataset ", dataset)
    accs2[dataset] = {}
    for method, method_res in methods.items():
        if not method in accs2[dataset]:
            accs2[dataset][method] = {}
        for seed, seed_res in method_res.items():
            for metric, metric_res in seed_res.items():
                if metric in ['median']:
                    continue
                #print(metric_res)
                hparam, _ = metric_res #GIVEN THE METRIC, THOSE ARE THE BEST HPARAMS
                idx = hparam.find("LR") #LR_0.00006250
                lr = hparam[idx+3:idx+13]
                mean_error = final_results[dataset][method][seed][hparam]["mean error"]
                #print(mean_error)
                acc = 100. - mean_error
                if metric in accs2[dataset][method]:
                    accs2[dataset][method][metric].append((lr,acc))
                else:
                    accs2[dataset][method][metric] = [(lr,acc)]
                        

mx = ["CONSISTENCY", "ENTROPY", "SOURCE_mean", "SMALL_mean_error", "mean error"]


for mm in mx:

    fig, axes = plt.subplots(1, num_datasets, figsize=(num_datasets * 6, 6), sharey=True)
    #plt.figure(dpi=100)  # Adjust DPI as needed

    idx = 0

    m_dict = {"CONSISTENCY" : "CON", "ENTROPY" : "ENT", 
            "SOURCE_mean" : "S_ACC", "cross_val" : "CROSS-ACC", 
            "SMALL_mean_error" : "100-RND", 
            "mean error" : "ORACLE"}
    idx = 0
  
    for dataset, results in accs.items():
        data = []
        for method in methods:
            res = results[method]
            res.sort(key=lambda x: x[0], reverse=True)
           
            data.extend([(lr, acc) for lr, acc in res])

    for dataset, results in accs2.items():
        data2 = []
        for method in methods:
            res = results[method][mm]
            res.sort(key=lambda x: x[0], reverse=True)
           
            data2.extend([(lr, acc) for lr, acc in res])
            
        scatterplot(x='Accuracy', y='LR', ax=axes[idx], data=pd.DataFrame(data, columns=['LR', 'Accuracy']), color='blue')
        scatterplot(x='Accuracy', y='LR', ax=axes[idx], data=pd.DataFrame(data2, columns=['LR', 'Accuracy']), color='red')
        if idx == 0:
            axes[idx].set_ylabel('LR')
        else:
            axes[idx].set_ylabel('')

        axes[idx].set_xlim([0, 75]) #TODO
        axes[idx].set_title("{}".format(dataset.upper()))
            #plt.title("{}".format(dataset))
            
        idx += 1
            # Show the plot
        
    #plt.gcf().set_dpi(300) 
    plt.tight_layout()    
    plt.savefig("plots/{}_acc_by_LR.png".format(mm), dpi=300)  # Adjust DPI as needed
    #plt.show()    