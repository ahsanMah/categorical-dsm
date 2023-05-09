#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
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
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


DATASET = sys.argv[1]

assert DATASET in ["census", "bank", "probe", "u2r", "cmc", "solar", "chess"]
print("==="*10, "Running baselines for:", DATASET, "==="*10)
# In[5]:


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from torchinfo import summary
from models.score_base import TabScoreModel

import seaborn as sns
import matplotlib as mpl

mpl.rc('figure', figsize=(10, 4), dpi=100)
sns.set_theme()


# In[6]:


from dataloader import get_dataset, build_tabular_ds
from torch.utils.data import DataLoader
from models.mutils import onehot_to_logit


# In[7]:


from configs import census_config as cfg

config = cfg.get_config()
config


# In[8]:


import runner
workdir = f"/workspace/categorical-dsm/results/{config.data.dataset}/"
workdir


# In[9]:


from ood_detection_helper import ood_metrics, auxiliary_model_analysis
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize


# In[10]:


def get_msma_results(workdir, ckpt_num=-1, seed=0):
    #TODO: USE PYOD MODELS!
    np.random.seed(42)
    workdir = os.path.join(workdir, f"seed_{seed}")
    denoise = config.msma.denoise
    ckpt_dir = os.path.join(workdir, "checkpoints")
    ckpts = sorted(os.listdir(ckpt_dir))
    ckpt = ckpts[ckpt_num]
    step = ckpt.split("-")[0]
    fname = os.path.join(
            workdir, "score_norms", f"{step}-{'denoise' if denoise else ''}-score_norms.npz"
        )
    with np.load(fname, allow_pickle=True) as npzfile:
        outdict = {k: npzfile[k].item() for k in npzfile.files}


    X_train = outdict["train"]["score_norms"]
    np.random.shuffle(X_train)
    X_val = outdict["val"]["score_norms"]
    X_train = np.concatenate((X_train[: len(X_val)], X_val))
    test_labels = outdict["test"]["labels"]
    X_test = outdict["test"]["score_norms"][test_labels == 0]
    X_ano = outdict["test"]["score_norms"][test_labels == 1]
    results = auxiliary_model_analysis(X_train, X_test, [X_ano],
                                    components_range=range(3,11,2),
                                    labels=["Train", "Inlier", "Outlier"])
    
    return results


# In[11]:


import warnings
warnings.filterwarnings("ignore")


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

hyp={
	 'input_dim':input_size,
	 'hidden1_dim':1024,
	 'hidden2_dim':512,
	 'hidden3_dim':256,
	 'zc_dim':2,
	 'emb_dim':128,
	 'n_gmm':2,
	 'dropout':0.5,
	 'lambda1':0.1,
	 'lambda2':0.005,
	 'lr' :1e-4,
	 'batch_size':512,
	 'epochs': 100,
	 'print_iter': 1,
	 'savestep_epoch': 1,
	 'save_dir': f'./workdir/baselines/dagmm/{config.data.dataset}',
	 # 'data_dir': '../dagmm-master/kdd_cup.npz',
	 'img_dir': f'./workdir/baselines/dagmm/{config.data.dataset}',
	 'ratio' : None,
    'patience_epochs': 5,
    'checkpoint' : "best"
}

# model definitions
dsvdd_clf = partial(DeepSVDD, hidden_neurons=[input_size, 1024, 512, 256], use_ae=True,
                   hidden_activation="leaky_relu", output_activation="leaky_relu",
                   optimizer="adam",verbose=1, epochs=100, batch_size=512)

model_dict = {'DAGMM': dagmm_test_runner, 'IForest': PYOD, 'ECOD': PYOD,  'DSVDD': dsvdd_clf, }

model_list = list(model_dict.keys())
model_results = defaultdict(list)

# seed for different folds
for idx in range(5):
    print(f"------- Starting run for seed {idx} -------")
    
    config.seed = 42 + idx
    
    train_ds, val_ds, test_ds = get_dataset(config, train_mode=False, return_logits=False,
                                        return_with_loader=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=hyp['batch_size'],
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
    
    for x,y in test_loader:
        X_test.append(x.numpy())
        y_labels.append(y.numpy())

    X_test = np.concatenate(X_test)
    y_labels = np.concatenate(y_labels)
    ano_ratio = sum(y_labels)/(len(y_train)+len(y_labels))
    
    # X_train.shape, X_test.shape, y_labels.shape
    
    for name, clf in model_dict.items():
        print(f"Started: {name}")
        start = time()
        
        if name == "DAGMM":
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
            score = out[:,1][:, None]
        elif name == "DSVDD":
            score = clf.decision_function(X_test)
        else:
            score = clf.predict_score(X_test)

        # evaluation
        results = ood_metrics(score[y_labels==0], score[y_labels==1], plot=False, verbose=False)
        results["seed"] = idx
        
        # save results
        model_results[name].append(results)
        print(f"Completed! Time Elapsed: {(time()-start):.2f}s")  
        
    print("----------------------------------------")


# In[15]:


# save the results
baseline_metrics = []

for m,r in model_results.items():
    df = pd.DataFrame(r) * 100
    df["model"] = m
    baseline_metrics.append(df)
baseline_metrics = pd.concat(baseline_metrics)
baseline_metrics.to_csv(f"results/{config.data.dataset}_baseline_metrics.csv")

baseline_metrics[["roc_auc", "ap", "model"]].groupby('model').describe()


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

