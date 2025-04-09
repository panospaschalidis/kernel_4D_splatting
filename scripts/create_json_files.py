import os, glob
import numpy as np
import random
import json
import argparse

# create dataset.json and metadata.json files. scene.json elements
# are not being emplpoyed, so wew are just incorporate hypernerf's
# scene.json so as not to hit in hyper_loader

if __name__=="__main__":

    parser = argparse.ArgumentParser("Create json files")
    parser.add_argument("--video_instance", type=str, default=None)

    args = parser.parse_args()
    seq_len = len(glob.glob(os.path.join(args.video_instance,"rgb","1x","*")))
    seq = np.arange(seq_len).tolist()

    data_dict = {}
    meta_dict = {}

    data_dict["count"] = seq_len
    data_dict["num_exemplars"] = seq_len
    data_dict["ids"] = ["frame_"+str(idx).zfill(6) for idx in seq]
    train_per = 0.7
    test_per = 0.3
    train_len = int(seq_len*0.7)
    test_len = int(seq_len*0.3)
    train_ids = random.sample(seq, train_len)
    # reform seq
    [seq.pop(idx) for idx in sorted(train_ids,reverse=True)]
    test_ids = random.sample(seq, int(test_len*(2/3)))
    [seq.pop(seq.index(idx)) for idx in sorted(test_ids,reverse=True)]
    val_ids = seq
    data_dict["train_ids"] = ["frame_"+str(idx).zfill(6) for idx in sorted(train_ids)]
    data_dict["test_ids"] = ["frame_"+str(idx).zfill(6) for idx in sorted(test_ids)]
    data_dict["val_ids"] = ["frame_"+str(idx).zfill(6) for idx in sorted(val_ids)]
    with open(f"{args.video_instance}/dataset.json", "w") as f:
        json.dump(data_dict, f, indent=1)

    for val,frame in enumerate(data_dict["ids"]):
        meta_dict[frame] = {"time_id": val, "warp_id": val, "appearance_id": val, "camera_id": 0}

    with open(f"{args.video_instance}/metadata.json", "w") as f:
        json.dump(meta_dict, f, indent=1)
