import sys
import pickle
import torch
import os
import numpy as np
import math

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

def single_gene_ablation(data, model, gene_keys, ordered_feature_masks):
    # data should be pre-filtered for BMI category and mse
    diffs_dict = {}
    for k in gene_keys:
        print(k)
        diffs = []
        mask = torch.tensor(ordered_feature_masks[k])
        for i in range(len(data)):
            og_inp = data[i].to(device)
            og_pheno = model(og_inp.float())
            new_inp = og_inp * mask
            new_pheno = model(new_inp.float())
            diff = (og_pheno - new_pheno).detach().cpu().numpy()
            diffs.append(diff)
        diffs_dict[k] = np.concatenate(diffs)
    # persist diffs dict
    return diffs_dict

def get_unsigned_means(diffs_dict):
    unsigned_means_dict = {}
    for key in diffs_dict.keys():
        unsigned_means_dict[key] = np.mean(np.absolute(diffs_dict[key]))
    return unsigned_means_dict

def one_gene_pairwise(data, ordered_feature_masks, start_gene, gene_set,
                      diffs_dict, model, save_path=None):
    # perturb given gene with comparison set and store perturbation results
    # comparison_set should be list of strings
    # data should be pre-filtered for phenotype category (if desired) and MSE
    pairs_dict = {}
    gene_mask = ordered_feature_masks[start_gene]
    g = 1
    for gene in gene_set:
        print("gene %i of %i" % (g, len(gene_set)), end='\r')
        key = start_gene + "_" + gene
        mask = ordered_feature_masks[gene]
        joint_mask = torch.tensor(gene_mask * mask).to(device)
        diff_diffs = []
        c = 0
        for i in range(len(data)):
            linear_diff = diffs_dict[start_gene][c] + diffs_dict[gene][c]
            og_inp = data[i].to(device)
            og_pheno = model(og_inp.float())
            new_inp = og_inp * joint_mask
            new_pheno = model(new_inp.float())
            pair_diff = (og_pheno - new_pheno).detach().cpu().numpy()
            diff_diffs.append(pair_diff - linear_diff)
            c += 1
        pairs_dict[key] = np.mean(np.absolute(diff_diffs))
        g += 1
        if save_path:
            pickle.dump(pairs_dict, open(os.path.join(save_path,gene,"_pairs.pkl"), "wb"))
    return pairs_dict

def pairwise_ablation(data, ordered_feature_masks, gene_set,
                      diffs_dict, model, save_path=None):
    # exhaustive search of comparison set
    pairs_dicts = []
    searched_genes = set()
    for start_gene in gene_set:
        print(start_gene)
        searched_genes.add(start_gene)
        comparison_set = [g for g in gene_set if g not in searched_genes]
        pd = one_gene_pairwise(data, ordered_feature_masks, start_gene, comparison_set, diffs_dict,
                          model, save_path)
        pairs_dicts.append(pd)
    return pairs_dicts

def check_overlap(gene1, gene2, gene_feature_masks):
    # check if two genes have SNPs in common
    mask1 = gene_feature_masks[gene1]
    mask2 = gene_feature_masks[gene2]
    set1 = set(np.where(mask1==0)[0])
    set2 = set(np.where(mask2==0)[0])
    if len(set1.intersection(set2)) > 0:
        return True
    else:
        return False

def check_second_degree_overlap(gene1, gene2, gene_feature_masks, comparison_set):
    # check if the two input genes have overlapping genes in common
    for gene3 in comparison_set:
        if (check_overlap(gene1, gene3, gene_feature_masks) and
            check_overlap(gene2, gene3, gene_feature_masks)):
            return True, gene3
    return False, None

def add_other_dict_keys(search_gene, dict_directory):
    files = os.listdir(dict_directory)
    og_path = dict_directory + search_gene + "_pairs_dict.pkl"
    og_dict = pickle.load(open(og_path, "rb"))
    for f in files:
        key = f.split("_")[0] + "_" + search_gene
        dict = pickle.load(open(dict_directory + f, "rb"))
        if key in dict.keys():
            new_key = search_gene + "_" + key.split("_")[0]
            og_dict[new_key] = dict[key]

def main(ordered_feature_masks_file, model_file, test_data_path, pairs_directory):
    ordered_feature_masks = pickle.load(open(ordered_feature_masks_file,"rb"))
    model = torch.load(model_file)
    model.to(device)
    print(model)
    test_samples = np.load(test_data_path)
    diffs_dict = single_gene_ablation(test_samples, model, ordered_feature_masks.keys(), ordered_feature_masks)
    means_dict = get_unsigned_means(diffs_dict)
    sorted_effects = sorted(means_dict.items(), key=lambda x:x[1], reverse=True)
    print("--INDIVIDUAL EFFECTS--\n")
    for key in sorted_effects.keys():
        print("{}: {}".format(key,sorted_effects[key]))
    pairs_dicts = pairwise_ablation(test_samples, ordered_feature_masks, ordered_feature_masks.keys(), diffs_dict, model, pairs_directory)
    for gene in ordered_feature_masks.keys():
        add_other_dict_keys(gene, pairs_directory)

if __name__ == "__main__":
    main(ordered_feature_masks_file=sys.argv[1], model_file=sys.argv[2], test_data_path=sys.argv[3], pairs_directory=sys.argv[4])
