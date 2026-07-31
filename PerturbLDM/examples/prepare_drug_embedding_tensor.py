import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util_diff import Condition_IDConverter


parser = argparse.ArgumentParser(
    description="Align a drug-embedding dictionary to PerturbLDM drug IDs."
)
parser.add_argument("--cond2id_path", required=True,
                    help="Path to processed/cond2id.json for the target dataset")
parser.add_argument("--embedding_pkl", required=True,
                    help="Pickle file containing either an embeddings dict or {'embeddings': dict}")
parser.add_argument("--output_path", required=True,
                    help="Output .pt path for the aligned drug-embedding tensor")
parser.add_argument("--alias_json", default=None,
                    help="Optional JSON mapping dataset drug names to embedding-dictionary drug names")
args = parser.parse_args()


with open(args.cond2id_path, "r") as f:
    cond2id = json.load(f)
condconvertor = Condition_IDConverter(cond2id.keys())

with open(args.embedding_pkl, "rb") as f:
    embedding_data = pickle.load(f)
drug_emb_dict = embedding_data.get("embeddings", embedding_data)

alias = {}
if args.alias_json is not None:
    with open(args.alias_json, "r") as f:
        alias = json.load(f)

drug_emb_list = []
missing_drugs = []
for ii in range(len(condconvertor.drug2id)):
    drug_name = condconvertor.id2drug[ii]
    embedding_name = drug_name if drug_name in drug_emb_dict else alias.get(drug_name)
    if embedding_name is None or embedding_name not in drug_emb_dict:
        missing_drugs.append(drug_name)
        continue
    drug_emb_list.append(drug_emb_dict[embedding_name])

if missing_drugs:
    raise ValueError(
        "Missing embeddings for drug names: " + ", ".join(missing_drugs)
    )

drug_emb_pretrain_np = np.vstack(drug_emb_list)
drug_emb_pretrain = torch.tensor(drug_emb_pretrain_np, dtype=torch.float32)

os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
torch.save(drug_emb_pretrain, args.output_path)
print(f"Saved aligned drug embedding tensor: {args.output_path}")
print(f"Tensor shape: {tuple(drug_emb_pretrain.shape)}")
