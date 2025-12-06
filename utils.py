import torch
import logging
import numpy as np
from datasets.imagenet_subsets import IMAGENET_D_MAPPING
import torch.nn as nn
import torch.nn.functional as F
import random
from augmentations.transforms_cotta import get_tta_transforms
from models.entropy_loss import EntropyLoss
from pytorch_metric_learning.distances import BatchedDistance, CosineSimilarity


logger = logging.getLogger(__name__)
epsilon = 2./255

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

#@torch.jit.script
def KL(logit1,logit2,reverse=False):
    if reverse:
        logit1, logit2 = logit2, logit1
    p1 = logit1.softmax(1)
    logp1 = logit1.log_softmax(1)
    logp2 = logit2.log_softmax(1) 
    return (p1*(logp1-logp2)).sum(1)


def split_results_by_domain(domain_dict, data, predictions):
    """
    Separate the labels and predictions by domain
    :param domain_dict: dictionary, where the keys are the domain names and the values are lists with pairs [[label1, prediction1], ...]
    :param data: list containing [images, labels, domains, ...]
    :param predictions: tensor containing the predictions of the model
    :return: updated result dict
    """

    labels, domains = data[1], data[2]
    assert predictions.shape[0] == labels.shape[0], "The batch size of predictions and labels does not match!"

    for i in range(labels.shape[0]):
        if domains[i] in domain_dict.keys():
            domain_dict[domains[i]].append([labels[i].item(), predictions[i].item()])
        else:
            domain_dict[domains[i]] = [[labels[i].item(), predictions[i].item()]]

    return domain_dict


def eval_domain_dict(domain_dict, domain_seq=None):
    """
    Print detailed results for each domain. This is useful for settings where the domains are mixed
    :param domain_dict: dictionary containing the labels and predictions for each domain
    :param domain_seq: if specified and the domains are contained in the domain dict, the results will be printed in this order
    """
    correct = []
    num_samples = []
    avg_error_domains = []
    dom_names = domain_seq if all([dname in domain_seq for dname in domain_dict.keys()]) else domain_dict.keys()
    logger.info(f"Splitting up the results by domain...")
    for key in dom_names:
        content = np.array(domain_dict[key])
        correct.append((content[:, 0] == content[:, 1]).sum())
        num_samples.append(content.shape[0])
        accuracy = correct[-1] / num_samples[-1]
        error = 1 - accuracy
        avg_error_domains.append(error)
        logger.info(f"{key:<20} error: {error:.2%}")
    logger.info(f"Average error across all domains: {sum(avg_error_domains) / len(avg_error_domains):.2%}")
    # The error across all samples differs if each domain contains different amounts of samples
    logger.info(f"Error over all samples: {1 - sum(correct) / sum(num_samples):.2%}")

def get_source_accuracy(model: torch.nn.Module,
                        source_data_loader: torch.utils.data.DataLoader,
                        device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_correct = 0
    with torch.no_grad():
        for i, data in enumerate(source_data_loader):
                #if i > 100:
                #    break
            imgs, labels = data[0], data[1]
            output = model([img.to(device) for img in imgs], disable_adaptation=True) if isinstance(imgs, list) else model(imgs.to(device), disable_adaptation=True)
            #this includes forward pass and model adaptation
            predictions = output.argmax(1)

            source_correct += (predictions == labels.to(device)).float().sum()

    source_accuracy = source_correct.item() / len(source_data_loader.dataset)
    return source_accuracy

def mask_out_self(sim_mat, start_idx, return_mask=False):
    num_rows, num_cols = sim_mat.shape
    mask = torch.ones(num_rows, num_cols, dtype=torch.bool)
    rows = torch.arange(num_rows)
    cols = rows + start_idx
    mask[rows, cols] = False
    sim_mat = sim_mat[mask].view(num_rows, num_cols - 1)
    if return_mask:
        return sim_mat, mask
    return sim_mat

def get_iter_fn(all_entropies, entropy_fn, T):
    def fn(sim_mat, s, *_):
        sim_mat = mask_out_self(sim_mat, s)
        sim_mat = F.softmax(sim_mat / T, dim=1)
        all_entropies.append(entropy_fn(sim_mat))

    return fn


def get_accuracy(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_name: str,
                 domain_name: str,
                 setting: str,
                 domain_dict: dict,
                 oracle_tests: int, #HOW MANY SAMPLES FOR THE EVALUATION
                 #source_data_loader: torch.utils.data.DataLoader = None,
                 device: torch.device = None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    correct = 0.
    small_correct = 0.
    small_accuracy = None
    small_total = 0
    tta_transforms = get_tta_transforms(dataset_name, soft=True)

    how_many_batches = len(data_loader)
    batches_list = [x + 1 for x in range(how_many_batches -1)] #JUST SKIP THE 1st batch, wait for the adaptation to happen
    random.shuffle(batches_list)
    batches_list = batches_list[0:oracle_tests]

    entropy_list = []
    consistency_list = []
    snd_list = []

    with torch.no_grad():
        for i, data in enumerate(data_loader):

            imgs, labels = data[0], data[1]
            output = model([img.to(device) for img in imgs]) if isinstance(imgs, list) else model(imgs.to(device))
            entropy_loss = softmax_entropy(output).mean(0).cpu().numpy()
            entropy_list.append(entropy_loss)
            #this includes forward pass and model adaptation
            predictions = output.argmax(1)
            if isinstance(imgs, list):
                imgs_aug = [tta_transforms(img) for img in imgs]
            else:    
                imgs_aug = tta_transforms(imgs)
            output_aug = model([img.to(device) for img in imgs_aug], disable_adaptation=True) if isinstance(imgs_aug, list) else model(imgs_aug.to(device), disable_adaptation=True)
        
            consistency_loss = KL(output.detach(), output_aug.detach(), reverse=False).mean(0).cpu().numpy()
            consistency_list.append(consistency_loss)

            T = 0.05
            entropy_fn = EntropyLoss(after_softmax=True, return_mean=False)
            dist_fn = BatchedDistance(CosineSimilarity(), batch_size=1024)
            all_entropies = []
            dist_fn.iter_fn = get_iter_fn(all_entropies, entropy_fn, T)
            #print("shapes ", imgs.shape, labels.shape)
            if labels.shape[0] == 1:
                #FOR MEMO, run augmentations of the same image. Not sure if this will work properly though
                augs = model.get_augs(imgs.to(device))
                output_aug = model([img.to(device) for img in augs], disable_adaptation=True) if isinstance(augs, list) else model(augs.to(device), disable_adaptation=True)
                dist_fn(output_aug)
            else:
                dist_fn(output)

            all_entropies = torch.cat(all_entropies, dim=0)
            snd = -torch.mean(all_entropies).item()
            snd_list.append(snd)

            if dataset_name == "imagenet_d" and domain_name != "none":
                mapping_vector = list(IMAGENET_D_MAPPING.values())
                predictions = torch.tensor([mapping_vector[pred] for pred in predictions], device=device)

            correct += (predictions == labels.to(device)).float().sum()
    
            if oracle_tests > 0:
                if i in batches_list:
                    small_correct += float(predictions[0] == labels[0].to(device))
                    small_total += 1

            if "mixed_domains" in setting and len(data) >= 3:
                domain_dict = split_results_by_domain(domain_dict, data, predictions)
    
    entropy_loss = np.mean(entropy_list)
    consistency_loss = np.mean(consistency_list)
    snd_loss = np.mean(snd_list)
    
    accuracy = correct.item() / len(data_loader.dataset)
    
    if oracle_tests > 0:
        print("small_total ", small_total)
        small_accuracy = small_correct / (small_total)
    results_dict = {"acc": accuracy,
                   #"source_acc": source_accuracy,
                   "entropy": entropy_loss,
                   "oracle_100": small_accuracy,
                   "consistency_loss" : consistency_loss,
                   "snd_loss" : snd_loss}
    return results_dict, domain_dict

def get_online_accuracy(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_name: str,
                 domain_name: str,
                 setting: str,
                 domain_dict: dict,
                 oracle_tests: int, #HOW MANY SAMPLES FOR THE EVALUATION
                 device: torch.device = None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    correct = 0.
    total = 0.
    small_correct = 0.
    small_accuracy = None
    small_total = 0
    tta_transforms = get_tta_transforms(dataset_name, soft=True)

    how_many_batches = len(data_loader)
    batches_list = [x + 1 for x in range(how_many_batches -1)] #JUST SKIP THE 1st batch, wait for the adaptation to happen
    random.shuffle(batches_list)
    batches_list = batches_list[0:oracle_tests]
    
    entropy_list = []
    consistency_list = []
    snd_list = []

    with torch.no_grad():
        for i, data in enumerate(data_loader):

            imgs, labels = data[0], data[1]
            output = model([img.to(device) for img in imgs]) if isinstance(imgs, list) else model(imgs.to(device))
            entropy_loss = softmax_entropy(output).mean(0).cpu().numpy()
            entropy_list.append(entropy_loss)
            #this includes forward pass and model adaptation
            predictions = output.argmax(1)
            if isinstance(imgs, list):
                imgs_aug = [tta_transforms(img) for img in imgs]
            else:    
                imgs_aug = tta_transforms(imgs)
            output_aug = model([img.to(device) for img in imgs_aug], disable_adaptation=True) if isinstance(imgs_aug, list) else model(imgs_aug.to(device), disable_adaptation=True)
        
            consistency_loss = KL(output.detach(), output_aug.detach(), reverse=False).mean(0).cpu().numpy()
            consistency_list.append(consistency_loss)

            T = 0.05
            entropy_fn = EntropyLoss(after_softmax=True, return_mean=False)
            dist_fn = BatchedDistance(CosineSimilarity(), batch_size=1024)
            all_entropies = []
            dist_fn.iter_fn = get_iter_fn(all_entropies, entropy_fn, T)
            if labels.shape[0] == 1:
                augs = model.get_augs(imgs.to(device))
                output_aug = model([img.to(device) for img in augs], disable_adaptation=True) if isinstance(augs, list) else model(augs.to(device), disable_adaptation=True)
                dist_fn(output_aug)
            else:
                dist_fn(output)

            all_entropies = torch.cat(all_entropies, dim=0)
            snd = -torch.mean(all_entropies).item()
            snd_list.append(snd)

            if dataset_name == "imagenet_d" and domain_name != "none":
                mapping_vector = list(IMAGENET_D_MAPPING.values())
                predictions = torch.tensor([mapping_vector[pred] for pred in predictions], device=device)

            curr_correct = int((predictions == labels.to(device)).float().sum())
            correct += (predictions == labels.to(device)).float().sum()
            total += labels.shape[0]
    
            if oracle_tests > 0:
                if i in batches_list:
                    small_correct += float(predictions[0] == labels[0].to(device))
                    #print(small_correct, (predictions[0] == labels[0].to(device)))
                    small_total += 1

            if "mixed_domains" in setting and len(data) >= 3:
                domain_dict = split_results_by_domain(domain_dict, data, predictions)

            entropy_loss = np.mean(entropy_list)
            consistency_loss = np.mean(consistency_list)
            snd_loss = np.mean(snd_list)
    
            accuracy = float (correct.item() / total)

            #logger.info(f"{cfg.CORRUPTION.DATASET} error % [{domain_name}{severity}][#samples={len(test_data_loader.dataset)}]: {err:.2%}")
            logger.info("batch id: {}".format(i))
            logger.info(f"ent_curr:{entropy_list[-1]:.2%}%, entropy_loss:{entropy_loss:.2%}%")
            logger.info(f"con_curr:{consistency_list[-1]:.2%}%, consistency_loss:{consistency_loss:.2%}%")
            logger.info(f"snd_curr:{snd_list[-1]:.2%}%, snd_loss:{snd_loss:.2%}%")
            logger.info(f"correct:{curr_correct}, accuracy:{accuracy:.2%}%")
    
    entropy_loss = np.mean(entropy_list)
    consistency_loss = np.mean(consistency_list)
    snd_loss = np.mean(snd_list)
    
    accuracy = correct.item() / len(data_loader.dataset)
    
    if oracle_tests > 0:
        print("small_total ", small_total)
        small_accuracy = small_correct / (small_total)
    results_dict = {"acc": accuracy,
                   #"source_acc": source_accuracy,
                   "entropy": entropy_loss,
                   "oracle_100": small_accuracy,
                   "consistency_loss" : consistency_loss,
                   "snd_loss" : snd_loss}
    return results_dict, domain_dict