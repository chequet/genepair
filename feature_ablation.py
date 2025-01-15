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

def get_unsigned_means(diffs_dict, means_dict_path):
    unsigned_means_dict = {}
    for key in diffs_dict.keys():
        unsigned_means_dict[key] = np.mean(np.absolute(diffs_dict[key]))
    pickle.dump(unsigned_means_dict, open(means_dict_path, "wb"))
    return unsigned_means_dict

def one_gene_pairwise(data, ordered_feature_masks, start_gene, gene_set,
                      diffs_dict, model, dict_directory, device, lin_mod=False):
    # perturb given gene with comparison set and store perturbation results
    # comparison_set should be list of strings
    # data should be pre-filtered for phenotype category (if desired) and MSE
    pairs_dict = {}
    if not lin_mod:
        dict_path = dict_directory + start_gene + "_pairs_dict.pkl"
    else:
        dict_path = dict_directory + "LIN_" + start_gene + "_pairs_dict.pkl"
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
            if lin_mod:
                og_inp = data[i].reshape(1, -1)
                og_pheno = model.predict(og_inp)
                new_inp = (og_inp * joint_mask).reshape(1, -1)
                new_pheno = model.predict(new_inp)
                pair_diff = (og_pheno - new_pheno)
            else:
                og_inp = data[i].to(device)
                og_pheno = model(og_inp.float())
                new_inp = og_inp * joint_mask
                new_pheno = model(new_inp.float())
                pair_diff = (og_pheno - new_pheno).detach().cpu().numpy()
            diff_diffs.append(pair_diff - linear_diff)
            c += 1
        pairs_dict[key] = np.mean(np.absolute(diff_diffs))
        g += 1
    print("writing pairs dict for %s to %s..." % (start_gene, dict_path))
    pickle.dump(pairs_dict, open(dict_path, "wb"))

def one_gene_parallel_pairwise(data, ordered_feature_masks, start_gene, gene_set,
                      diffs_dict, model, dict_directory, n_cpus, device, lin_mod=False):
    # divide gene set up into n_cpus batches
    miniset_size = int(math.ceil(len(gene_set)/n_cpus))
    print("miniset size: %i"%miniset_size)
    minisets = []
    for i in range(0,len(gene_set),miniset_size):
        minisets.append(gene_set[i:i+miniset_size])
    procs = []
    for miniset in minisets:
        proc=Process(target=one_gene_pairwise, args=(data, ordered_feature_masks, start_gene, miniset,
                      diffs_dict, model, dict_directory, device, lin_mod,))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()

def pairwise_ablation(data, ordered_feature_masks, gene_set,
                      diffs_dict, stop_gene, model, dict_directory, device, lin_mod=False, parallel=False):
    # exhaustive search of comparison set
    searched_genes = set()
    for start_gene in gene_set:
        print(start_gene)
        if start_gene == stop_gene:
            print("reached stop gene: %s"%stop_gene)
            return True
        searched_genes.add(start_gene)
        comparison_set = [g for g in gene_set if g not in searched_genes]
        if parallel:
            one_gene_parallel_pairwise(data, ordered_feature_masks, start_gene, comparison_set, diffs_dict,
                              model, dict_directory, N_CPUs, device, lin_mod)
        else:
            one_gene_pairwise(data, ordered_feature_masks, start_gene, comparison_set, diffs_dict,
                          model, dict_directory, device, lin_mod)
    return True

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

def main(ordered_feature_masks_file, model_file):
    ordered_feature_masks = pickle.load(open(ordered_feature_masks_file,"rb"))
    model = torch.load(model_file)
    model.to(device)
    print(model)
    test_samples = np.load('data/X_tst.npy')
    gene_keys = list(ordered_feature_masks.keys())
    # filter for BMI category, MSE
    diffs_dict = pickle.load(open("../diffs_dicts/obese12diffs.pkl","rb"))
    unsigned_means_dict = get_unsigned_means(diffs_dict, "../diffs_dicts/obese12means.pkl")
    # sorted_unsigned_lin = sorted(lin_means.items(), key=lambda x: x[1], reverse=True)
    sorted_unsigned = sorted(unsigned_means_dict.items(), key=lambda x:x[1], reverse=True)
    # exhaustive search!
    print("beginning pairwise ablation...")
    genes = [tup[0] for tup in sorted_unsigned]
    pairwise_ablation(X_data, ordered_feature_masks, genes, diffs_dict, stop_gene, model,
                      "../diffs_dicts/", device, lin_mod=linmod, parallel=False)

if __name__ == "__main__":
    main()
