import os
import logging
import numpy as np
import torch

from models.model import get_model
from datasets.data_loading import get_test_loader, get_source_loader
from conf import cfg, load_cfg_from_args, get_num_classes, get_domain_sequence, adaptation_method_lookup


logger = logging.getLogger(__name__)


def evaluate(description):
    load_cfg_from_args(description)
    valid_settings = ["reset_each_shift",           # reset the model state after the adaptation to a domain
                      "continual",                  # train on sequence of domain shifts without knowing when a shift occurs
                      "gradual",                    # sequence of gradually increasing / decreasing domain shifts
                      "mixed_domains",              # consecutive test samples are likely to originate from different domains
                      "correlated",                 # sorted by class label
                      "mixed_domains_correlated",   # mixed domains + sorted by class label
                      "gradual_correlated",         # gradual domain shifts + sorted by class label
                      "reset_each_shift_correlated"
                      ]
    assert cfg.SETTING in valid_settings, f"The setting '{cfg.SETTING}' is not supported! Choose from: {valid_settings}"

    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)
    base_model = get_model(cfg, num_classes)

    # setup test-time adaptation method
    model = eval(f'{adaptation_method_lookup(cfg.MODEL.ADAPTATION)}')(cfg=cfg, model=base_model, num_classes=num_classes)
    logger.info(f"Successfully prepared test-time adaptation method: {cfg.MODEL.ADAPTATION.upper()}")
    logger.info("CORRUPTION DATASET - {}".format(cfg.CORRUPTION.DATASET))
    # get the test sequence containing the corruptions or domain names
    logger.info("corruption type {}".format(cfg.CORRUPTION.TYPE))
    if cfg.CORRUPTION.DATASET in {"domainnet126"}:
        # extract the domain sequence for a specific checkpoint.
        dom_names_all = get_domain_sequence(ckpt_path=cfg.CKPT_PATH)
    elif cfg.CORRUPTION.DATASET in {"imagenet_d", "imagenet_d109"} and not cfg.CORRUPTION.TYPE[0]:
        # dom_names_all = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        dom_names_all = ["clipart", "infograph", "painting", "real", "sketch"]
    else:
        dom_names_all = cfg.CORRUPTION.TYPE
    logger.info(f"Using the following domain sequence: {dom_names_all}")

    # prevent iterating multiple times over the same data in the mixed_domains setting
    dom_names_loop = ["mixed"] if "mixed_domains" in cfg.SETTING else dom_names_all

    # setup the severities for the gradual setting
    if "gradual" in cfg.SETTING and cfg.CORRUPTION.DATASET in {"cifar10_c", "cifar100_c", "imagenet_c"} and len(cfg.CORRUPTION.SEVERITY) == 1:
        severities = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        logger.info(f"Using the following severity sequence for each domain: {severities}")
    else:
        severities = cfg.CORRUPTION.SEVERITY

    num_tests = len(dom_names_loop) * len(severities)
    ORACLE_NUM = 100
    oracle_per_test = ORACLE_NUM // num_tests + 1

    _, src_loader = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET,
                                           root_dir=cfg.DATA_DIR, adaptation=cfg.MODEL.ADAPTATION,
                                           batch_size=128, #USED FOR TESTING-ONLY, LARGER BATCH SIZE 
                                           train_split=False, #USE VALIDATION PART OF THE SOURCE SET
                                           ckpt_path=cfg.CKPT_PATH, #only for {"domainnet126", "office31", "visda"}
                                           workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()))
    
    
    print("source dataeset samples = ", len(src_loader.dataset))

    import time
    start = time.time()
    num_tests = len(dom_names_loop) * len(severities)

    for xxx in range(int(cfg.REPEAT)):
        logger.info("ROUND {} out of {}".format(xxx, cfg.REPEAT))
        for i_dom, domain_name in enumerate(dom_names_loop):
            if (i_dom == 0 and xxx == 0) or "reset_each_shift" in cfg.SETTING:
                try:
                    model.reset()
                    logger.info("resetting model")
                except:
                    logger.warning("Error. Not resetting model")
            else:
                logger.warning("not resetting model")

            for severity in severities:
                test_data_loader = get_test_loader(setting=cfg.SETTING,
                                                adaptation=cfg.MODEL.ADAPTATION,
                                                dataset_name=cfg.CORRUPTION.DATASET,
                                                root_dir=cfg.DATA_DIR,
                                                domain_name=domain_name,
                                                severity=severity,
                                                num_examples=cfg.CORRUPTION.NUM_EX,
                                                rng_seed=cfg.RNG_SEED,
                                                domain_names_all=dom_names_all,
                                                alpha_dirichlet=cfg.TEST.ALPHA_DIRICHLET,
                                                batch_size=cfg.TEST.BATCH_SIZE,
                                                shuffle=False,
                                                workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()))

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                with torch.no_grad():
                    for i, data in enumerate(test_data_loader):
                        imgs, labels = data[0], data[1]
                        output = model([img.to(device) for img in imgs]) if isinstance(imgs, list) else model(imgs.to(device))

            
    end = time.time()
    logger.info("final results")
    
    logger.info(f"wall time {(end-start)}")

if __name__ == '__main__':
    evaluate('"Evaluation.')
