import torch
from torchvision import models
from torch import nn
from data import GazeModelDataset, CalibrationDataset
import cv2 as cv
import h5py

torch.autograd.set_detect_anomaly(True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# feature extractor for eye and face
feature_extractor = models.vgg16(pretrained=True)
for param in feature_extractor.parameters():
    param.requires_grad = False
feature_extractor.eval()

feature_extractor.to(device)


# model for subjectwiseembedding
class SubjectWiseEmbedding(nn.Module):
    def __init__(self, input_features, out_features, hidden_units):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=out_features),
        )

    def forward(self, subject_id):
        return self.linear_layer_stack(subject_id)


# model for gaze
class GazeModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        """Initializes all required hyperparameters for a multi-class classification model.

        Args:
            input_features (int): Number of input features to the model.
            out_features (int): Number of output features of the model
              (how many classes there are).
            hidden_units (int): Number of hidden units between layers, default 8.
        """
        super().__init__()
        self.lin1 = torch.nn.Linear(input_features, hidden_units)
        self.relu = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(hidden_units, output_features)

        # Initialize queries and pref_vecs properly
        # self.register_buffer('queries', torch.empty(0, input_features + 6))  # Adjust size as needed
        # self.register_buffer('pref_vecs', torch.empty(0, input_features))
        self.queries = torch.tensor([])
        self.pref_vecs = torch.tensor([])

    def forward(self, input, pref_vector, store_queries=False):
        face_features = feature_extractor(input["face"])
        eye_features = feature_extractor(input["eye"])

        temp = torch.cat((face_features, eye_features, input["head_rot_mat"], input["origin"]), axis=1)

        query = torch.cat((temp, input["gaze"]), axis=1)

        # need to store for calibration
        if store_queries == True:
            # lets store the queries for calibration model
            self.queries = torch.cat((self.queries, query.detach().cpu()), axis=0)
            self.pref_vecs = torch.cat((self.pref_vecs, pref_vector.detach().cpu()), axis=0)

        x = torch.cat((temp, pref_vector), axis=1)
        # used to solve the inplace modification error, reference: https://discuss.pytorch.org/t/runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation-torch-floattensor-64-1-which-is-output-0-of-asstridedbackward0-is-at-version-3-expected-version-2-instead-hint-the-backtrace-further-a/171826/7
        y1 = torch.nn.functional.linear(x, self.lin1.weight.clone(), self.lin1.bias)
        y2 = self.relu(y1)
        y3 = torch.nn.functional.linear(y2, self.lin2.weight.clone(), self.lin2.bias)

        return y3


# some testing experiments to make sure that dimensions are known
if __name__ == '__main__':
    hdf_files = ["outputs_pgte/MPIIGaze1.h5"]
    for file in hdf_files:
        with h5py.File(file, 'r') as f:
            id = 0
            for person_id, group in f.items():
                print('')
                print('Processing %s/%s' % (file, person_id))
                pgte_dataset = GazeModelDataset(group)
                from torch.utils.data import DataLoader

                train_dataloader_custom = DataLoader(dataset=pgte_dataset,  # use custom created train Dataset
                                                     batch_size=2,  # how many samples per batch?
                                                     num_workers=0,
                                                     # how many subprocesses to use for data loading? (higher = more)
                                                     shuffle=True)  # shuffle the data?
                out = next(iter(train_dataloader_custom))
                for key, val in out.items():
                    print(key)
                    print(val.shape)
                    if key == "side":
                        print(val)
                subject_wise_embedding = SubjectWiseEmbedding(1, 6, 32)
                gaze_model = GazeModel(2011, 3, 2048)
                print(out["subject_id"].unsqueeze(axis=1))
                subject_id = out["subject_id"].unsqueeze(axis=1)
                pref_vec = subject_wise_embedding(out["subject_id"].unsqueeze(axis=1).type(torch.float32))
                output = gaze_model(out, pref_vec)
                queries = gaze_model.queries
                pref_vecs = gaze_model.pref_vecs

                calibration_dataset = CalibrationDataset(pref_vec, queries, 1)
                calibration_dataloader = DataLoader(dataset=calibration_dataset, batch_size=2, num_workers=0,
                                                    shuffle=True)
                out = next(iter(calibration_dataloader))
                print(out[0].shape)
                print(out[1].shape)
                a = 1 / 0

                # preference_vector = torch.ones((1,6))
                # face_features = feature_extractor(out["face"])
                # print(face_features.shape)
                # eye_features = feature_extractor(out["eye"])
                # print(eye_features.shape)
                # x = torch.cat((face_features, eye_features, out["head_rot_mat"], out["origin"], preference_vector), axis = 1)

                print(output.shape)

                id += 1
