import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import models
import cv2 as cv
import h5py

debug = 0


# dataset for gaze
class GazeModelDataset(Dataset):
    """Dataset from HDF5 archives formed of 'groups' of specific persons."""

    def __init__(self, group):
        num_entries = next(iter(group.values())).shape[0]

        # format from the preprocess file
        self.numLen = num_entries
        self.eyes = group["eye"]
        self.faces = group["face"]
        self.head_rot_mats = group["head_rot_matrix"]
        self.origins = group["origin"]
        self.sides = group["side"]
        self.subject_ids = group["subject_id"]
        self.gazes = group["gaze"]


    def __len__(self):
        return self.numLen

    def close_hdf(self):
        if self.hdf is not None:
            self.hdf.close()
            self.hdf = None

    # preprocess function adopted from faze model
    def preprocess_image(self, image):
        if debug == 1:
            cv.imshow("original", image)
            cv.waitKey(0)

        ycrcb = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv.equalizeHist(ycrcb[:, :, 0])
        image = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2RGB)
        if debug == 1:
            cv.imshow("preprocessed", image)
            cv.waitKey(0)
        #hwc -> chw, torch accepts chws
        image = np.transpose(image, [2, 0, 1])  # Colour image
        image = 2.0 * image / 255.0 - 1


        return image

    def preprocess_entry(self, entry):
        for key, val in entry.items():
            if isinstance(val, np.ndarray):
                entry[key] = torch.from_numpy(val.astype(np.float32))
            elif isinstance(val, int):
                # NOTE: maybe ints should be signed and 32-bits sometimes
                entry[key] = torch.tensor(val, dtype=torch.int16, requires_grad=False)
        return entry

    def __getitem__(self, idx):
        head_rot_vec = np.array([np.arcsin(self.head_rot_mats[idx][1, 2]),
                  np.arctan2(self.head_rot_mats[idx][0, 2], self.head_rot_mats[idx][2, 2])])
        ret = {
            "eye" : self.preprocess_image(np.array(self.eyes[idx])),
            "face" : self.preprocess_image(np.array(self.faces[idx])),
            "head_rot_mat" : head_rot_vec,
            "origin" : self.origins[idx].reshape((3)),
            "side" : self.sides[idx],
            "subject_id" : self.subject_ids[idx],
            "gaze" : self.gazes[idx]
        }
        ret = self.preprocess_entry(ret)
        return ret


# calibration dataset
class CalibrationDataset(Dataset):

    def __init__(self, pref_vec, queries, s):
        n = queries.shape[0]
        m = pref_vec.shape[0]
        assert n > s , "expecting number of queries to be of multiple of s"

        l = n // s

        # drop the last tail queries
        self.queries = queries[:l*s]


        self.queries = self.queries.view(l, s, queries.shape[1])
        # repeat the same vector n times as it will be same given subject_id
        self.pref_vecs = pref_vec.repeat(l, 1)

    def __len__(self):
        return self.queries.shape[0]

    def __getitem__(self, idx):
        return self.queries[idx], self.pref_vecs[idx]


# testing code to be familiar with the dimensions
if __name__ == '__main__':
    hdf_files = ["outputs_pgte/MPIIGaze1.h5"]
    for file in hdf_files:
        with h5py.File(file, 'r') as f:
            id = 0
            for person_id, group in f.items():
                print('')
                print('Processing %s/%s' % (file, person_id))
                pgte_dataset = GazeModelDataset( group)
                from torch.utils.data import DataLoader

                train_dataloader_custom = DataLoader(dataset=pgte_dataset,  # use custom created train Dataset
                                                     batch_size=5,  # how many samples per batch?
                                                     num_workers=0,
                                                     # how many subprocesses to use for data loading? (higher = more)
                                                     shuffle=True)  # shuffle the data?
                out = next(iter(train_dataloader_custom))
                for key, val in out.items():
                    print(key)
                    print(val.shape)
                    if key == "side":
                        print(val)

                id+=1