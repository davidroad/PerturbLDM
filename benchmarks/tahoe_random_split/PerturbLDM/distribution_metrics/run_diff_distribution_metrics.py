import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0", help="CUDA device(s) to use, e.g., 0 or '0,1'")
parser.add_argument("--diff_dir", default="", help="diff_dir")
parser.add_argument("--dataset_id", default="Random_7_3_Jun6", help="dataset label recorded in outputs")
parser.add_argument("--data_root", required=True, help="Processed Tahoe split directory containing collection/ and processed/")

args = parser.parse_args()
dataset_id = args.dataset_id
print(dataset_id)

diff_dir = args.diff_dir
if not os.path.isdir(diff_dir):
    raise ValueError(f'Cannot find diff_dir: {diff_dir}')

save_dir = diff_dir
print(diff_dir)


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

print('cuda ', os.environ['CUDA_VISIBLE_DEVICES'])


import pandas as pd
import numpy as np
import sys
import torch
torch.cuda.is_available()

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler, RandomSampler
import time
import json
import scanpy as sc
import anndata as ad
from scipy.sparse import issparse
from torch.utils.data import Dataset
from typing import Optional, Callable, Dict, List
import time
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


import json
import torch
import random
from collections import defaultdict

temppath = os.path.join(diff_dir, 'all_cf_exprs_diffU.npy')
all_cf_exprs = np.load(temppath) # (num_conditions, num_cells, num_genes)

temppath = os.path.join(diff_dir, 'test_allcondnames.json')
diff_inference_condnames = json.load(open(temppath, 'r'))


datadir = os.path.abspath(args.data_root)
if not os.path.isdir(datadir):
    raise ValueError(f"Cannot find data_root: {datadir}")
processed_datadir = os.path.join(datadir, "processed")
collection_datadir = os.path.join(datadir, "collection")

test_adata = sc.read_h5ad(os.path.join(collection_datadir, 'test_adata.h5ad'))
test_df = pd.read_csv(os.path.join(collection_datadir, 'test_metadf.csv'))
condname_test_list = list(test_df['CondID'])

import random
from collections import defaultdict

groups_test = defaultdict(list)
for idx, name in enumerate(condname_test_list):
    groups_test[name].append(idx)



from final_distribution_metrics import distributional_similarity_metrics
# distributional_similarity_metrics(X_real, X_pred,

import time
start_time = time.time()



cond_result = {}
for ii, condi in enumerate(diff_inference_condnames):
    cf_expr = all_cf_exprs[ii]
    gt_expr = test_adata.X[groups_test[condi]].toarray()
    print(condi)   
    dist_metrics = distributional_similarity_metrics(gt_expr, cf_expr)
    dist_metrics['CondID'] = condi
    cond_result[condi] = dist_metrics
    # if ii > 10:
        # break


end_time = time.time()
print(f"Total time taken: {round((end_time - start_time)/60, 2)} minutes")

final_df000 = pd.DataFrame({kk:vv for kk,vv in cond_result.items()}).T
final_df000.to_csv(os.path.join(save_dir, f'all_dist_metrics_percondition_diffU.csv'))

