import torch

from gaze_model import GazeModel, SubjectWiseEmbedding

from train_eval import train_calib_model

device = "cuda" if torch.cuda.is_available() else "cpu"
import h5py
import os



if __name__ == '__main__':
    # dataset hdf files, grabcapture file can also be added
    hdf_files = ["/content/drive/MyDrive/outputs_pgte/MPIIGaze1.h5"]

    k_folds = 2
    for file in hdf_files:
        with h5py.File(file, 'r') as f:
            subject_id = 0
            for person_id, group in f.items():
                print('')
                print('Processing %s/%s' % (file, person_id))
                # Preprocess some datasets
                output_dir = './checkpoints/{}'.format(person_id)

                queries = torch.load("{}/queries.pth".format(output_dir))

                # get the pref_vec from a trained subjectwise_embedding model
                model_path = "{}/subjectwise_model-fold-{}.pth".format(output_dir, k_folds - 1)
                state_dict = torch.load(model_path)
                subject_wise_embedding = SubjectWiseEmbedding(1, 6, 32).to(device)
                subject_wise_embedding.load_state_dict(state_dict)

                # for testing, ignore
                # queries = torch.randn((72, 2008))
                subject_id = torch.tensor(group["subject_id"][0]).unsqueeze(0).unsqueeze(0)
                subject_id = subject_id.type(torch.float32).to(device)
                pref_vec = subject_wise_embedding(subject_id)
                pref_vec = pref_vec.squeeze(0)

                # run the training for calibration
                results = train_calib_model(queries=queries, pref_vecs=pref_vec, output_dir=output_dir, num_epochs=2,
                                            train_batch_size=8, eval_batch_size=8, person_id=person_id, s=16)
                subject_id += 1