import pickle as pkl
import sys

default_cp_configs = {
    "do_cp": False,
    "cp_times": 1,
    "p_cp": 1,
    # "p_cp": 0.5,

    "do_inter_cp": False,
    "p_inter_cp": 1,
    "do_match": False,

    "do_elastic": False,
    "elastic_deform_alpha": (0., 900.),
    "elastic_deform_sigma": (9., 13.),
    "p_eldef": 1,
    # "p_eldef": 0.2,

    "do_scaling": False,
    "scale_range": (0.85, 1.25),
    "p_scale": 1,
    # "p_scale": 0.2,

    "do_rotation": False,
    "degree": 180,
    "p_rot": 1,
    # "p_rot": 0.2,

    "do_gamma": False,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 1,
    # "p_gamma": 0.3,

    "do_mirror": False,
    "p_mirror": 1,
    # "p_mirror": 0.8,

    "do_blurring": False,
    "blur_sigma": (1, 5),
    "p_blur": 1,
    # "p_blur": 0.1,

    # Not Implemented
    # "do_additive_brightness": False,
    # "additive_brightness_p_per_sample": 0.15,
    # "additive_brightness_p_per_channel": 0.5,
    # "additive_brightness_mu": 0.0,
    # "additive_brightness_sigma": 0.1,
}

def prepare_pkl(file_path):
    with open(file_path,'wb') as handle:
        pkl.dump(cp_configs, handle)

if __name__ == "__main__":
    file_path = sys.argv[1]
    prepare_pkl(file_path)