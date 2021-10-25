#!/usr/bin/env python
# coding=utf-8
# Author: Yao
# Mail: zhangyao215@mails.ucas.ac.cn

import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from medpy import metric
from collections import OrderedDict
from scipy.ndimage.measurements import label
from metrics import *

join = os.path.join

def print_case_results(filename, label, scores):
    print('{} label: {}'.format(filename, label))
    for metric in metric_list:
        print(metric, scores[filename][label][metric])

def print_summary_results(label, scores):
    print('{}'.format(label))
    for metric in metric_list:
        print('mean_'+metric, scores[label]['mean_'+metric])
        print('std_'+metric, scores[label]['std_'+metric])
        print('len: {}'.format(len(scores[label][metric])))

if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_path")
    parser.add_argument("gt_path")
    parser.add_argument("n_label", type=int)

    args = parser.parse_args()

    pred_path = args.pred_path
    gt_path = args.gt_path
    n_label = args.n_label
    
    
    pred_path = join('./nnUNet_base/nnUNet_results/nnUNet', args.model, args.task, 'nnUNetTrainer__nnUNetPlans', 'fold_{}'.format(args.fold) if len(args.fold)==1 else 'all', 'validation')
    gt_path = join('./nnUNet_base/nnUNet_raw_splitted', args.task, 'labelsTr')
    '''
    
    task = 'Task000_KiTSbaseline'
    model = 'nnUNetTrainerV2_ImgDAObjAllinter10percents__nnUNetPlansv2.1'
    fold = 'fold_0'
    pred_path = join('../data/RESULTS_FOLDER/nnUNet/3d_fullres', task, model, fold, 'validation_raw')
    gt_path = join('../data/nnUNet_raw_data', task, 'labelsTr')
    n_label = 3
          
    # metric_list = ['dice', 'jaccard', 'hausdorff_distance_95', 'avg_surface_distance_symmetric', 'precision', 'recall']
    metric_list = ['dice', 'hausdorff_distance_95']
    # metric_list = ['dice']
    label_range = range(1, n_label) # start with 1 to exclude background

    exclude_list = ['00002', '00020', '00045', '00094', '00115', '00124']
    # exclude_list = []

    file_list = np.sort(os.listdir(pred_path))
    file_list = [x for x in file_list if x.endswith('nii.gz') and x.split('.')[0].split('_')[1] not in exclude_list]

    print('files len:', len(file_list))

    scores = {}
    for i in label_range:
        scores[i] = {}
        for metric in metric_list:
            scores[i][metric] = []
    
    #################### aggregate results of each case
    for filename in file_list:
        scores[filename] = {}
        affine = nib.load(join(gt_path, filename)).affine

        ori_pred_volume = nib.load(join(pred_path, filename)).get_fdata()
        ori_gt_volume = nib.load(join(gt_path, filename)).get_fdata()
    
        for i in label_range:
            scores[filename][i] = {}
            
            confusion_matrix = ConfusionMatrix(ori_pred_volume>=i, ori_gt_volume>=i)

            # label does not exist
            #  np.sum(ori_gt_volume[ori_gt_volume == i]) == 0:
            #     continue

            for metric in metric_list:
                scores[filename][i][metric] = eval(metric)(ori_pred_volume>=i, ori_gt_volume>=i, confusion_matrix, nan_for_nonexisting=False)
                scores[i][metric].append(scores[filename][i][metric])

            print_case_results(filename, i, scores)
    ##########################################
    
    ######### aggregate results as a summary
    for i in label_range:
        for metric in metric_list:
            scores[i]['mean_'+metric] = np.mean(scores[i][metric])
            scores[i]['std_'+metric] = np.std(scores[i][metric])
    
        print_summary_results(i, scores)
    ###################################
        
    ########## save as csv
    header = []
    for i in label_range:
        for metric in metric_list:
            header.append(metric+'_for_label'+str(i))
            
    rows = []
    for k in file_list:
        row = []
        for i in label_range:
            if len(scores[k][i].values()) > 0:
                row += scores[k][i].values()
            else:
                row += [0] * len(metric_list)
        rows.append(row)

    row = []
    for i in label_range:
        for metric in metric_list:
            row.append(scores[i]['mean_'+metric])
            # row.append(scores[i]['std_'+metric])
    rows.append(row)
    
    row = []
    for i in label_range:
        for metric in metric_list:
            # row.append(scores[i]['mean_'+metric])
            row.append(scores[i]['std_'+metric])
    rows.append(row)
            
    subject_ids = file_list + ['mean', 'std']
    
    df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
    df.to_csv(join(pred_path, 'results.csv'))
    ########################

