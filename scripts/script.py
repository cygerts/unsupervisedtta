import os
import numpy as np
import copy


results = "output_collas"

datasets = ["cifar100_c", "imagenet_c", "domainnet126", "imagenet_r", "imagenet_v2"] #DEFAULT SCENARIO

s_acc = {"cifar100_c" : 53.55,
         "imagenet_c" : 17.97,
         "domainnet126": 54.71,
         "imagenet_r" : 36.17,
         "imagenet_v2" : 58.72}

methods = ["tent", "eata", "sar", "rmt_SF", "adacontrast", "memo", "lame"] #ALL

print_summary = True
add_cross_val_metric = True
print_ranks = False
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
            print("parsing ", curr_dir)
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

                    #print(lines[-6]) #TODO, add to metrics
                    print("reading ", file_to_read)
                    for idx, metric in enumerate(measures):

                        if not params_key in final_results[dataset][method][seed]:
                            final_results[dataset][method][seed][params_key] = {}
                        if not params_key in h_params_keys[method]:
                            h_params_keys[method][seed].append(params_key)
                        try:
                            assert(metric in lines[-idx -1])
                        except:

                            print(method, "err ", params_key, metric, lines[-idx -1])
                            #for x in range(0,6):
                            #    print(lines[-x])
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
            

avg_measures = {} #averaged per dataset
agg_measures = {} #method x dataset x metrics x target_accuracy

all_measures = measures + ["median"]
if add_cross_val_metric:
    all_measures = all_measures + ["cross_val"]
er = 0
for dataset, methods in res_measures.items():
    if not dataset in agg_measures:
        agg_measures[dataset] = {}
    for method, method_res in methods.items():
        per_metric = {}
        print(dataset, method)
        if not method in avg_measures:
            avg_measures[method] = {}
        if not method in agg_measures[dataset]:
            agg_measures[dataset][method] = {}
        
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
              
                if not metric in per_metric:
                    per_metric[metric] = [acc]
                else:
                    per_metric[metric].append(acc)
           
        for metric in all_measures:
            curr = per_metric[metric]
            print(metric, np.mean(curr), np.std(curr), curr)
            if not metric in avg_measures[method]:
                avg_measures[method][metric] = curr
            else:
                avg_measures[method][metric].extend(curr)
            if not metric in agg_measures[dataset][method]:
                agg_measures[dataset][method][metric] = copy.deepcopy(curr)

            else:
                raise("should not happen")


print("\n Average results \n")
#print(avg_measures)
for method, method_res in avg_measures.items():
    for metric, metric_results in method_res.items():
        print(method, len(metric_results), metric, np.mean(metric_results))


print("\n Measures pairwise comparison \n")

print_mes = ["SOURCE_mean", "cross_val", "ENTROPY", "CONSISTENCY", "SND", "SMALL_mean_error",  "median", "mean error"]

if print_ranks:
    for row_measure in print_mes:
        for col_measure in print_mes:
            if row_measure == col_measure:
                continue
            row_wins = 0
            col_wins = 0
            for dataset in datasets:
                for method in methods:
                    
                    row_res = np.mean(agg_measures[dataset][method][row_measure])
                    col_res = np.mean(agg_measures[dataset][method][col_measure])
                    if row_res > col_res:
                        row_wins += 1
                    elif row_res < col_res:
                        col_wins += 1
                    else:
                        row_wins += 1
                        col_wins += 1
                    
            print("{} wins with {} is {} out of {}".format(row_measure, col_measure, row_wins, (row_wins + col_wins)))
                     

#RESULTS BY MODEL SELECTION STRATEGIES
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style="whitegrid")  # Set the style of the plot
data = []

num_datasets = len(datasets)

accs = {} #dataset x method. All methods aggregated

for dataset, methods in res_measures.items():
    print("Dataset ", dataset)
    accs[dataset] = {}
    for method, method_res in methods.items():
        for seed, seed_res in method_res.items():
            for metric, tuples in seed_res.items():
                if 'median' in tuples[0]:
                    continue

                hparams = tuples[0]
                try:
                    mean_error = final_results[dataset][method][seed][hparams]["mean error"]
                except:
                    print("no value at {} {} {}".format(dataset, method, seed))
                #assert(len(mean_error) == 1)
                acc = 100. - np.mean(mean_error)
                if metric in accs[dataset]:
                    accs[dataset][metric].append(acc)
                else:
                    accs[dataset][metric] = [acc]

fig, axes = plt.subplots(1, num_datasets, figsize=(num_datasets * 6, 6), sharey=True)
#plt.figure(dpi=100)  # Adjust DPI as needed

idx = 0

m_dict = {"CONSISTENCY" : "CON", "ENTROPY" : "ENT", 
        "SOURCE_mean" : "S-ACC", "cross_val" : "C-ACC", 
        "SMALL_mean_error" : "100-RND", 
        "mean error" : "ORACLE", "SND":"SND"}
idx = 0
if add_cross_val_metric:
    mx = ["SND", "CONSISTENCY", "ENTROPY", "SOURCE_mean", "cross_val", "SMALL_mean_error", "mean error"]
else:
    mx = ["SND", "CONSISTENCY", "ENTROPY", "SOURCE_mean", "SMALL_mean_error", "mean error"]

fsize = 20
for dataset, results in accs.items():
    data = []
    for metric in mx:
        try:
            res = results[metric]
        except:
            print("HERE", metric, dataset)
            raise()
        #all_res = mean_res[dataset][method]
        data.extend([(m_dict[metric], accuracy) for accuracy in res])
    
    #sns.violinplot(x=[item[0] for item in data], y=[item[1] for item in data])
    sns.boxplot(x='Accuracy', y='Method', ax=axes[idx], data=pd.DataFrame(data, columns=['Method', 'Accuracy']), color='blue')
    axes[idx].set_xlabel('Accuracy [%]',fontsize=fsize)
    y_values = [4.5, 4.5]  # Indexes corresponding to Method A and Method C
    #axes[idx].plot([-5, 100, np.nan, -5, -5], [y_values[0], y_values[1]], color='red', linestyle='--', linewidth=2)
    axes[idx].hlines(y=y_values, xmin=-10, xmax=100, color='green', linestyle='--', linewidth=2)
    x_values = [s_acc[dataset], s_acc[dataset]]
    axes[idx].vlines(x=x_values, ymin=-0.5, ymax=6.5, color='red', linestyle='--', linewidth=2, label="Source")
   
    if idx == 0:
        axes[idx].set_ylabel('Selection strategy',fontsize=fsize)
        axes[idx].legend(fontsize=fsize)
    else:
        axes[idx].set_ylabel('')

    axes[idx].set_xlim([0, 75])#TODO
    ddd = dataset.upper().replace("_", "-")
    axes[idx].set_title("{}".format(ddd),fontsize=fsize)
    axes[idx].tick_params(axis='both', labelsize=fsize)
    #plt.title("{}".format(dataset))
    
    idx += 1
    # Show the plot
    
#plt.gcf().set_dpi(300) 
plt.tight_layout()    
plt.savefig("plots/selection.png", dpi=300)  # Adjust DPI as needed
#plt.show()

#RESULTS BY TTA METHODS

sns.set(style="whitegrid")  # Set the style of the plot
data = []

num_datasets = len(datasets)

fig, axes = plt.subplots(1, num_datasets, figsize=(num_datasets * 6, 6), sharey=True)
#plt.figure(dpi=100)  # Adjust DPI as needed
    
print(axes)
idx = 0

#final_results = {}  #dataset x method x seed x params_key x metrics

for dataset, method_res in final_results.items():
    data = []
    print("Dataset ", dataset)
    for method, seed_res in method_res.items():
        method_res = []
        for seed, params_results in seed_res.items():
            for params, metric_results in params_results.items():
                err = metric_results["mean error"]
                acc = 100. - err
                method_res.append(acc)
    #           print(dataset, method, seed, params, acc)
        data.extend([(method.upper(), accuracy) for accuracy in method_res])
      
    sns.boxplot(x='Accuracy', y='Method', ax=axes[idx], data=pd.DataFrame(data, columns=['Method', 'Accuracy']), color='blue')
    x_values = [s_acc[dataset], s_acc[dataset]]
    axes[idx].vlines(x=x_values, ymin=-0.5, ymax=6.5, color='red', linestyle='--', linewidth=2, label="Source")
    if idx == 0:
        axes[idx].set_ylabel('Method',fontsize=fsize)
    else:
        axes[idx].set_ylabel('',fontsize=fsize)
    axes[idx].set_xlabel('Accuracy [%]',fontsize=fsize)
    axes[idx].set_xlim([0, 70]) #TODO
    ddd = dataset.upper().replace("_", "-")
    axes[idx].set_title("{}".format(ddd),fontsize=fsize)
    
    idx += 1
    # Show the plot
    
#plt.gcf().set_dpi(150) 
plt.tight_layout()    
plt.savefig("plots/methods.png", dpi=300)  # Adjust DPI as needed
#plt.show()
    
