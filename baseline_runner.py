#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys

module_path = os.path.abspath("/workspace/categorical-dsm/")
if module_path not in sys.path:
    sys.path.append(module_path)
os.chdir(module_path)

adbench_path = "/workspace/categorical-dsm/adbench_minimal/"
if adbench_path not in sys.path:
    sys.path.append(adbench_path)
# In[2]:


import tensorflow as tf
physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from ood_detection_helper import ood_metrics, auxiliary_model_analysis
from dataloader import get_dataset
import warnings
warnings.filterwarnings("ignore")

from configs import (
    census_config,
    solar_config,
    chess_config,
    bank_config,
    probe_config,
    u2r_config,
    cmc_config,
)

DATASET = sys.argv[1]
config_map = {
    "census": census_config,
    "solar": solar_config,
    "chess": chess_config,
    "bank": bank_config,
    "probe": probe_config,
    "u2r": u2r_config,
    "cmc": cmc_config,
}
assert DATASET in config_map
cfg = config_map[DATASET]
config = cfg.get_config()
print("===" * 10, "Running baselines for:", config.data.dataset, "===" * 10)



workdir = f"/workspace/categorical-dsm/results/{config.data.dataset}/"
workdir


# In[12]:


input_size = sum(config.data.categories) + config.data.numerical_features

from time import time

# from adbench_minimal.baseline.DAGMM.run import DAGMM
from DAGMM_pytorch.train import train as dagmm_train_runner

# from DAGMM_pytorch.test import decision_function as dagmm_clf
from DAGMM_pytorch.test import main as dagmm_test_runner
from pyod.models.deep_svdd import DeepSVDD
from collections import defaultdict
from baseline.PyOD import PYOD
from functools import partial
from torch.utils.data import DataLoader


hyp = {
    "input_dim": input_size,
    "hidden1_dim": 1024,
    "hidden2_dim": 512,
    "hidden3_dim": 256,
    "zc_dim": 2,
    "emb_dim": 128,
    "n_gmm": 2,
    "dropout": 0.5,
    "lambda1": 0.1,
    "lambda2": 0.005,
    "lr": 1e-4,
    "batch_size": 256,
    "epochs": 200,
    "print_iter": 1,
    "savestep_epoch": 1,
    "save_dir": f"./workdir/baselines/dagmm/{config.data.dataset}",
    # 'data_dir': '../dagmm-master/kdd_cup.npz',
    "img_dir": f"./workdir/baselines/dagmm/{config.data.dataset}",
    "ratio": None,
    "patience_epochs": 10,
    "checkpoint": "best",
    "return_logits":False,
}

# Mostly taken from KDDCUP-Rev config from original DAGMM paper
# Most other configs are unstable and frequently result in NaNs during training
if config.data.dataset in ["probe", "u2r"]:
    hyp["hidden1_dim"] = 120
    hyp["hidden2_dim"] = 60
    hyp["hidden3_dim"] = 30
    hyp["emb_dim"] = 10
    hyp["n_gmm"] = 1
    hyp["zc_dim"] = 1
    hyp["batch_size"] = 4096
    hyp["epochs"] = 100
    hyp["return_logits"] = True

if config.data.dataset in ["chess"]:
    hyp["batch_size"] = 4096

if config.data.dataset in ["bank"]:
    hyp["hidden1_dim"] = 64
    hyp["hidden2_dim"] = 32
    hyp["hidden3_dim"] = 16
    hyp["emb_dim"] = 10
    hyp["batch_size"] = 4096
    hyp["lr"] = 1e-5
    # hyp["return_logits"] = True
    # hyp["lambda2"] = 0.0001 # From https://github.com/tnakae/DAGMM
    
if config.data.dataset in ["census"]:
    hyp["hidden1_dim"] = 256
    hyp["hidden2_dim"] = 128
    hyp["hidden3_dim"] = 64
    hyp["emb_dim"] = 10
    # hyp["zc_dim"] = 1
    # hyp["return_logits"] = True
    hyp["batch_size"] = 4096



# model definitions
dsvdd_clf = partial(
    DeepSVDD,
    hidden_neurons=[input_size, 1024, 512, 256],
    use_ae=False,
    hidden_activation="swish",
    optimizer="adam",
    verbose=1,
    epochs=1000,
    batch_size=512,
)

model_dict = {
    "DAGMM": dagmm_test_runner,
    "IForest": PYOD,
    "ECOD": PYOD,
    "DSVDD": dsvdd_clf,
}

model_list = list(model_dict.keys())
model_results = defaultdict(list)

# seed for different folds
for idx in range(5):
    print(f"------- Starting run for seed {idx} -------")

    config.seed = 42 + idx
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    torch.manual_seed(config.seed)

    train_ds, val_ds, test_ds = get_dataset(
        config, train_mode=False, return_logits=hyp["return_logits"], return_with_loader=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=hyp["batch_size"],
        num_workers=2,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=2048,
        num_workers=2,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=2048,
        num_workers=2,
        shuffle=False,
    )

    X_val = np.concatenate([x[0].numpy() for x in val_loader])
    X_train_ = np.concatenate([x[0].numpy() for x in train_loader])
    X_train = np.concatenate((X_train_, X_val))
    y_train = np.zeros(X_train.shape[0])

    y_labels = []
    X_test = []

    for x, y in test_loader:
        X_test.append(x.numpy())
        y_labels.append(y.numpy())

    X_test = np.concatenate(X_test)
    y_labels = np.concatenate(y_labels)
    ano_ratio = sum(y_labels) / (len(y_train) + len(y_labels))

    # X_train.shape, X_test.shape, y_labels.shape

    for name, clf in model_dict.items():
        print(f"Started: {name}")
        start = time()

        if name == "DAGMM":
            # DAGMM has a bug where it produces NaNs at the last seed
            # We tested many models but they were always unstable for this dataset + seed
            # We decided to use the previous seed which would prduce a slightly bias result
            if idx == 4 and config.data.dataset == "probe":
                hyp["save_dir"] = f"./workdir/baselines/dagmm/{config.data.dataset}/seed_{idx-1}"
            else:
                hyp["save_dir"] = f"./workdir/baselines/dagmm/{config.data.dataset}/seed_{idx}"
                dagmm_train_runner(hyp, train_loader, val_loader)
        elif name == "DSVDD":
            clf = clf(random_state=config.seed, contamination=ano_ratio)
            clf = clf.fit(X=X_train, y=None)
        else:
            clf = clf(seed=config.seed, model_name=name)
            clf = clf.fit(X_train=X_train, y_train=y_train, ratio=ano_ratio)

        # output predicted anomaly score on testing set
        if name == "DAGMM":
            out = clf(hyp, test_loader)
            score = out[:, 1][:, None]
        elif name == "DSVDD":
            score = clf.decision_function(X_test)
        else:
            score = clf.predict_score(X_test)

        # evaluation
        results = ood_metrics(
            score[y_labels == 0], score[y_labels == 1], plot=False, verbose=False
        )
        results["seed"] = idx

        # save results
        model_results[name].append(results)
        print(f"Completed! Time Elapsed: {(time()-start):.2f}s")

    print("----------------------------------------")


# In[15]:


# save the results
baseline_metrics = []

for m, r in model_results.items():
    df = pd.DataFrame(r) * 100
    df["model"] = m
    baseline_metrics.append(df)
baseline_metrics = pd.concat(baseline_metrics)
baseline_metrics.to_csv(f"results/{config.data.dataset}_baseline_metrics.csv")

baseline_metrics[["roc_auc", "ap", "model"]].groupby("model").describe()


# In[ ]:


# baseline_metrics = pd.read_csv(f"results/{config.data.dataset}_baseline_metrics.csv", index_col=0)
# baseline_metrics[["roc_auc", "ap", "model"]].groupby('model').describe()


# # In[ ]:


# all_metrics = []
# for i in range(5):
#     msma_results = get_msma_results(workdir, seed=i)
#     all_metrics.append(msma_results)


# # In[ ]:


# gmm_metrics  = pd.concat(m["GMM"]["metrics"].reset_index(drop=True) for m in all_metrics
#                         ).reset_index(drop=True)
# gmm_metrics['seed'] = np.arange(5)
# gmm_metrics['model'] = "MSMA-GMM"
# gmm_metrics.describe()


# # In[ ]:


# kd_metrics  = pd.concat(m["KD"]["metrics"].reset_index(drop=True) for m in all_metrics
#                        ).reset_index(drop=True)
# kd_metrics['seed'] = np.arange(5)
# kd_metrics['model'] = "MSMA-KD"
# kd_metrics


# # In[ ]:


# kd_metrics.describe()


# # In[ ]:


# # save the results
# df_metrics = pd.concat([gmm_metrics, kd_metrics])

# for m,r in model_results.items():
#     df = pd.DataFrame(r) * 100
#     df["model"] = m
#     df_metrics = pd.concat((df_metrics, df))

# df_metrics[["roc_auc", "ap", "model"]].groupby('model').describe()


# # In[ ]:


# df_metrics.to_csv(f"results/{config.data.dataset}_final_metrics.csv")


# # In[ ]:


# df_stats = df_metrics.groupby('model').describe()

# for metric in ["ap", "roc_auc"]:
#     latex_str = [metric]
#     df = df_stats.loc[["IForest","ECOD","DAGMM","DSVDD","MSMA-GMM"], metric]
#     best =  df["mean"].max()
#     for m in df[["mean", "std"]].values:
#         _str = f"{m[0]:.2f} \pm~{m[1]:.2f}"
#         if np.isclose(m[0], best):
#             _str = "$\\mathbf{"+_str+"}$"
#         latex_str.append(_str)
#     latex_str = " & ".join(latex_str)
#     print(latex_str)


# # In[ ]:


# df_melt = df_metrics.drop(columns="seed").melt(id_vars="model", var_name="metric")
# df_melt


# # In[ ]:


# sns.catplot(data=df_melt.query("metric=='roc_auc'"), x="metric", y="value", hue="model", kind="bar")


# # In[ ]:


# sns.catplot(data=df_melt.query("metric=='ap'"), x="metric", y="value", hue="model", kind="bar")
