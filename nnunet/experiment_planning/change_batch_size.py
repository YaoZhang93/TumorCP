from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np

if __name__ == '__main__':
    # input_file = '/home/fabian/data/nnUNet_preprocessed/Task004_Hippocampus/nnUNetPlansv2.1_plans_3D.pkl'
    # output_file = '/home/fabian/data/nnUNet_preprocessed/Task004_Hippocampus/nnUNetPlansv2.1_LISA_plans_3D.pkl'
    # a = load_pickle(input_file)
    # a['plans_per_stage'][0]['batch_size'] = int(np.floor(6 / 9 * a['plans_per_stage'][0]['batch_size']))
    # save_pickle(a, output_file)
    
    input_file = '../../data/nnUNet_preprocessed/Task100_LiTSbaseline/nnUNetPlansv2.1_plans_3D.pkl'
    output_file = '../../data/nnUNet_preprocessed/Task100_LiTSbaseline/nnUNetPlansv2.1_plans_3D.pkl'
    a = load_pickle(input_file)
    print(a['plans_per_stage'])
    # a['plans_per_stage'][0]['batch_size'] = int(np.floor(6 / 9 * a['plans_per_stage'][0]['batch_size']))
    a['plans_per_stage'][0]['patch_size'] = np.array([128, 128, 128])
    a['plans_per_stage'][1]['patch_size'] = np.array([128, 128, 128])
    a['plans_per_stage'][0]['num_pool_per_axis'] = np.array([5, 5, 5])
    a['plans_per_stage'][1]['num_pool_per_axis'] = np.array([5, 5, 5])
    a['plans_per_stage'][0]['pool_op_kernel_sizes'] = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    a['plans_per_stage'][1]['pool_op_kernel_sizes'] = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    a['plans_per_stage'][0]['conv_kernel_sizes'] = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    a['plans_per_stage'][1]['conv_kernel_sizes'] = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    save_pickle(a, output_file)