#    Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import glob
import shutil
from collections import OrderedDict
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import label


def export_segmentations(indir, outdir):
    niftis = subfiles(indir, suffix='nii.gz', join=False)
    for n in niftis:
        identifier = str(n.split("_")[-1][:-7])
        outfname = join(outdir, "test-segmentation-%s.nii" % identifier)
        img = sitk.ReadImage(join(indir, n))
        sitk.WriteImage(img, outfname)


def export_segmentations_postprocess(indir, outdir):
    maybe_mkdir_p(outdir)
    niftis = subfiles(indir, suffix='nii.gz', join=False)
    for n in niftis:
        print("\n", n)
        identifier = str(n.split("_")[-1][:-7])
        outfname = join(outdir, "test-segmentation-%s.nii" % identifier)
        img = sitk.ReadImage(join(indir, n))
        img_npy = sitk.GetArrayFromImage(img)
        lmap, num_objects = label((img_npy > 0).astype(int))
        sizes = []
        for o in range(1, num_objects + 1):
            sizes.append((lmap == o).sum())
        mx = np.argmax(sizes) + 1
        print(sizes)
        img_npy[lmap != mx] = 0
        img_new = sitk.GetImageFromArray(img_npy)
        img_new.CopyInformation(img)
        sitk.WriteImage(img_new, outfname)


if __name__ == "__main__":
    processing_kernels = 8
    train_id_range = range(0, 150)
    test_id_range = range(150, 210)
    train_dir = "/dfsdata2/zhongcheng_data/zhangyao_data/DB/kits19/ori_data/"
    train_aug_dir = "/dfsdata2/zhongcheng_data/zhangyao_data/DB/dataaug/ori_data/kits_flip_rotate_scaling/"
    test_dir = "/dfsdata2/zhongcheng_data/zhangyao_data/DB/dataaug/ori_data/kits_flip_rotate_scaling/"

    output_folder = "../data/nnUNet_raw_data/Task001_KiTSrigid/"
    img_dir = join(output_folder, "imagesTr")
    lab_dir = join(output_folder, "labelsTr")
    img_dir_te = join(output_folder, "imagesTs")
    # os.makedirs(img_dir)
    # os.makedirs(lab_dir)
    # os.makedirs(img_dir_te)


    def load_save_train(args):
        data_file, seg_file = args
        data_pat_id = data_file.split("/")[-1]
        seg_pat_id = seg_file.split("/")[-1]
        print(data_pat_id)
        print(seg_pat_id)

        print(sitk.ReadImage(data_file).GetSize())
        # img_itk = sitk.ReadImage(data_file)
        # sitk.WriteImage(img_itk, join(img_dir, pat_id + "_0000.nii.gz"))
        shutil.copy(data_file, join(img_dir, data_pat_id))
        print('train data copy \nfrom', data_file, '\nto', join(img_dir, data_pat_id))

        print(sitk.ReadImage(seg_file).GetSize())
        # img_itk = sitk.ReadImage(seg_file)
        # sitk.WriteImage(img_itk, join(lab_dir, pat_id + ".nii.gz"))
        shutil.copy(seg_file, join(lab_dir, seg_pat_id))
        print('train seg copy \nfrom', seg_file, '\nto', join(lab_dir, seg_pat_id))
        # assert(sitk.ReadImage(data_file).GetSize() == sitk.ReadImage(seg_file).GetSize())
        return seg_pat_id.replace(".nii.gz", "")

    def load_save_test(args):
        data_file = args
        pat_id = data_file.split("/")[-1]
        print(pat_id)

        # img_itk = sitk.ReadImage(data_file)
        # sitk.WriteImage(img_itk, join(img_dir_te, pat_id + "_0000.nii.gz"))
        shutil.copy(data_file, join(img_dir_te, pat_id))
        print('test data copy \nfrom', data_file, '\nto', join(img_dir_te, pat_id))
        return pat_id.replace("_" + pat_id.split("_")[-1], "")

    # nii_files_tr_data = subfiles(train_dir, True, "volume", "nii", True)
    # nii_files_tr_seg = subfiles(train_dir, True, "segmen", "nii", True)
    # nii_files_ts = subfiles(test_dir, True, "test-volume", "nii", True)

    nii_files_tr_data = [x for x in np.sort(glob.glob(train_aug_dir+'data_volume/*.nii.gz')) if int(x.split('/')[-1].split('.')[0].split('_')[1].split('aug')[0]) in train_id_range] + [x for x in np.sort(glob.glob(train_dir+'data_volume/*.nii.gz')) if int(x.split('/')[-1].split('.')[0].split('_')[1]) in train_id_range]
    print('train data:', len(nii_files_tr_data))
    nii_files_tr_seg = [x for x in np.sort(glob.glob(train_aug_dir+'seg_volume/*.nii.gz')) if int(x.split('/')[-1].split('.')[0].split('_')[1].split('aug')[0]) in train_id_range] + [x for x in np.sort(glob.glob(train_dir+'seg_volume/*.nii.gz')) if int(x.split('/')[-1].split('.')[0].split('_')[1]) in train_id_range]
    print('train label:', len(nii_files_tr_seg))
    nii_files_ts = [x for x in np.sort(glob.glob(train_dir+'data_volume/*.nii.gz')) if int(x.split('/')[-1].split('.')[0].split('_')[1].split('aug')[0]) in test_id_range]
    print('test data:', len(nii_files_ts))

    p = Pool(processing_kernels)
    train_ids = p.map(load_save_train, zip(nii_files_tr_data, nii_files_tr_seg))
    test_ids = p.map(load_save_test, nii_files_ts)
    p.close()
    p.join()

    json_dict = OrderedDict()
    json_dict['name'] = "KiTS"
    json_dict['description'] = "KiTS"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT"
    }

    json_dict['labels'] = {
        "0": "background",
        "1": "kidney",
        "2": "tumor"
    }

    json_dict['numTraining'] = len(train_ids)
    json_dict['numTest'] = len(test_ids)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in train_ids]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_ids]

    with open(os.path.join(output_folder, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)
