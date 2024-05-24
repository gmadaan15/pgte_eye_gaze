import torch
from torch.utils.data import DataLoader, random_split

from sklearn.model_selection import KFold, train_test_split
from data import GazeModelDataset, CalibrationDataset
from gaze_model import GazeModel, SubjectWiseEmbedding
from huber_loss import HuberLoss
from tqdm import tqdm
from calibration_loss import CalibrationLoss
from calibration_model import CalibrationModel

device = "cuda" if torch.cuda.is_available() else "cpu"
import h5py
import os


def reset_weights(m):
    '''
      Try resetting model weights to avoid
      weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def to_device(entry):
    for key, val in entry.items():
        entry[key] = entry[key].to(device)


def train_epoch(gaze_model: torch.nn.Module,
                subject_wise_embedding: torch.nn.Module,
                trainloader: torch.utils.data.DataLoader,
                gaze_loss_function: torch.nn.Module,
                subject_wise_embedding_loss_function: torch.nn.Module,
                gaze_model_optimizer: torch.optim.Optimizer,
                subject_wise_embedding_optimizer: torch.optim.Optimizer,
                gaze_scheduler: torch.optim.lr_scheduler,
                subject_wise_embedding_scheduler: torch.optim.lr_scheduler,
                fold, epoch, num_epochs_cosine, intial_num_epochs, optimise_subject_wise=False, k_folds=5
                ):
    print("training for epoch:{}".format(epoch))
    # Put model in train mode
    gaze_model.train()
    subject_wise_embedding.train()

    # Set current loss value
    current_gaze_loss = 0.0
    current_subject_embedding_loss = 0.0

    # Training with cosine decay schedule
    for i, data in enumerate(trainloader, 0):
        # print(data)
        to_device(data)
        # print(data)
        gts = data["gaze"].to(device)

        # Zero the gradients
        gaze_model_optimizer.zero_grad()
        if optimise_subject_wise == True:
            subject_wise_embedding_optimizer.zero_grad()

        # Perform forward pass
        subject_id = data["subject_id"].unsqueeze(axis=1)
        pref_vec = subject_wise_embedding(subject_id.type(torch.float32).to(device))
        # if optimise_subject_wise == True:
        # pref_vec = pref_vec.detach()

        # if its the last fold, store the queries for calibration model
        # print("************* storing ******************{}%%%{}&&&{}=={}".format(fold,
        # k_folds,epoch, num_epochs_cosine + intial_num_epochs))
        if fold == k_folds - 1 and epoch == num_epochs_cosine + intial_num_epochs:
            # print("************* storing ******************{}".format(epoch))
            outputs = gaze_model(data, pref_vec, store_queries=True)
        else:
            outputs = gaze_model(data, pref_vec, False)

        # Compute loss
        gaze_loss = gaze_loss_function(outputs, gts)

        if optimise_subject_wise == True:
            gaze_loss.backward(retain_graph=True)
        else:
            # Perform backward pass
            gaze_loss.backward()

        # Perform optimization
        gaze_model_optimizer.step()

        current_gaze_loss += gaze_loss.item()

        # paper says it moves the gradients after 40 epochs for subjectwiseembedding
        if optimise_subject_wise == True:
            # llar for subjectwise embedding
            subject_wise_embedding_loss = subject_wise_embedding_loss_function(outputs, gts)
            # subject_wise_embedding.requires_grad = True
            subject_wise_embedding_loss.backward()
            subject_wise_embedding_optimizer.step()
            current_subject_embedding_loss += subject_wise_embedding_loss.item()

    # also applied scheduler after 40 epochs in the paper
    if optimise_subject_wise == True:
        gaze_scheduler.step()  # Update learning rate
        subject_wise_embedding_scheduler.step()

    if optimise_subject_wise == True:
        print(
            f'Epoch {epoch}/{num_epochs_cosine + intial_num_epochs}, Gaze Loss: {current_gaze_loss / len(trainloader)}, '
            f'SubjectWise Loss:{current_subject_embedding_loss / len(trainloader)}')
    else:
        print(f'Epoch {epoch}/{intial_num_epochs + num_epochs_cosine}, Gaze Loss: {current_gaze_loss}')


def eval_epoch(
        gaze_model: torch.nn.Module,
        subject_wise_embedding: torch.nn.Module,
        testloader: torch.utils.data.DataLoader,
        gaze_loss_function: torch.nn.Module,
        subject_wise_embedding_loss_function: torch.nn.Module,
        fold, epoch, num_epochs_cosine, intial_num_epochs, optimise_subject_wise=False, k_folds=5
):
    print("evaluating for epoch: {}".format(epoch))
    # Put model in train mode
    gaze_model.eval()
    subject_wise_embedding.eval()

    # Set current loss value
    current_gaze_loss = 0.0
    current_subject_embedding_loss = 0.0
    with torch.no_grad():
        # Training with cosine decay schedule
        for i, data in enumerate(testloader, 0):
            to_device(data)
            gts = data["gaze"].to(device)

            # Perform forward pass
            subject_id = data["subject_id"].unsqueeze(axis=1)
            pref_vec = subject_wise_embedding(subject_id.type(torch.float32).to(device))

            outputs = gaze_model(data, pref_vec, False)

            # Compute loss
            gaze_loss = gaze_loss_function(outputs, gts)

            current_gaze_loss += gaze_loss.item()

            if optimise_subject_wise == True:
                subject_wise_embedding_loss = subject_wise_embedding_loss_function(outputs, gts)
                current_subject_embedding_loss += subject_wise_embedding_loss.item()

        if optimise_subject_wise == True:
            print(
                f'Epoch {epoch}/{num_epochs_cosine + intial_num_epochs}, Gaze Loss: {current_gaze_loss / len(testloader)}, '
                f'SubjectWise Loss:{current_subject_embedding_loss / len(testloader)}')
        else:
            print(f'Epoch {epoch}/{intial_num_epochs + num_epochs_cosine}, Gaze Loss: {current_gaze_loss}')


def train_gaze_model(group, output_dir, k_folds=5, intial_num_epochs=40, num_epochs_cosine=40, train_batch_size=16,
                     eval_batch_size=16, person_id="p00"):
    # Configuration options
    # hardcoded values are directly taken from paper
    gaze_loss_function = HuberLoss(delta=1.5)
    subject_wise_embedding_loss_function = torch.nn.MSELoss()

    # For fold results
    results = {}

    # Set fixed random number seed
    torch.manual_seed(42)

    # Prepare MNIST dataset by concatenating Train/Test part; we split later.
    dataset = GazeModelDataset(group)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # return queries for the calibration model
    queries = None
    pref_vecs = None

    # subject id

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=train_batch_size, sampler=train_subsampler)
        evalloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=eval_batch_size, sampler=test_subsampler)

        # Init the neural network
        # 2011 is experimentally found, rest are choosen with intuition
        subject_wise_embedding = SubjectWiseEmbedding(1, 6, 32).to(device)
        gaze_model = GazeModel(2011, 3, 2048).to(device)
        gaze_model.apply(reset_weights)
        subject_wise_embedding.apply(reset_weights)

        # values are taken from paper
        # Initialize optimizer
        gaze_model_optimizer = torch.optim.Adam(gaze_model.parameters(), lr=3 * 1e-4, betas=(0.9, 0.999), eps=1e-07,
                                                weight_decay=0)
        # l2 regularisation for subjectwise embedding and this will be stepped after 40 epochs with lr = 10^(-4)
        subject_wise_embedding_optimizer = torch.optim.Adam(subject_wise_embedding.parameters(), lr=1e-4,
                                                            betas=(0.9, 0.999),
                                                            eps=1e-07, weight_decay=0.01)

        # Run the training loop for defined number of epochs
        for epoch in tqdm(range(1, intial_num_epochs + 1)):
            # Print epoch
            print(f'Starting epoch {epoch}')

            train_epoch(gaze_model,
                        subject_wise_embedding,
                        trainloader,
                        gaze_loss_function,
                        subject_wise_embedding_loss_function,
                        gaze_model_optimizer,
                        subject_wise_embedding_optimizer,
                        None,
                        None,
                        fold, epoch, num_epochs_cosine, intial_num_epochs, optimise_subject_wise=False, k_folds=k_folds
                        )

            eval_epoch(gaze_model,
                       subject_wise_embedding,
                       evalloader,
                       gaze_loss_function,
                       subject_wise_embedding_loss_function,
                       fold, epoch, num_epochs_cosine, intial_num_epochs, optimise_subject_wise=False, k_folds=k_folds)

        # Adjust learning rate to 10^-4 and use cosine decay schedule
        gaze_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(gaze_model_optimizer, T_max=num_epochs_cosine,
                                                                    eta_min=0.0001)
        subject_wise_embedding_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(subject_wise_embedding_optimizer,
                                                                                      T_max=num_epochs_cosine,
                                                                                      eta_min=0.0001)

        # after 40 epochs, train with different settings and also optimise subjectwise
        for epoch in tqdm(range(intial_num_epochs + 1, intial_num_epochs + num_epochs_cosine + 1)):
            # Print epoch
            print(f'Starting epoch {epoch}')

            train_epoch(gaze_model,
                        subject_wise_embedding,
                        trainloader,
                        gaze_loss_function,
                        subject_wise_embedding_loss_function,
                        gaze_model_optimizer,
                        subject_wise_embedding_optimizer,
                        gaze_scheduler,
                        subject_wise_embedding_scheduler,
                        fold, epoch, num_epochs_cosine, intial_num_epochs, optimise_subject_wise=True, k_folds=k_folds
                        )

            eval_epoch(gaze_model,
                       subject_wise_embedding,
                       evalloader,
                       gaze_loss_function,
                       subject_wise_embedding_loss_function,
                       fold, epoch, num_epochs_cosine, intial_num_epochs, optimise_subject_wise=True, k_folds=k_folds)

        # if this is the last fold, just save the queries and pref_vecs for the calibration model to train.
        if fold == k_folds - 1:
            queries = gaze_model.queries
            pref_vecs = gaze_model.pref_vecs

        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Saving the model
        save_path = f'./{output_dir}/gaze_model-fold-{fold}.pth'
        torch.save(gaze_model.state_dict(), save_path)
        save_path = f'./{output_dir}/subjectwise_model-fold-{fold}.pth'
        torch.save(subject_wise_embedding.state_dict(), save_path)

    return queries, pref_vecs


def train_calib_model(queries, pref_vecs, output_dir, num_epochs=1, train_batch_size=16, eval_batch_size=16,
                      person_id="p00", s=16):
    def train_epoch(model: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    loss_fn: torch.nn.Module,
                    optimizer: torch.optim.Optimizer):
        # Put model in train mode
        model.train()

        # Setup train loss and train accuracy values
        train_loss = 0

        # Loop through data loader data batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
            # print(X.shape)

            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate  and accumulate loss
            loss = loss_fn(y_pred, y)
            # print(loss.item())

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward(retain_graph=True)

            # 5. Optimizer step
            optimizer.step()

            train_loss += loss.item()
            print(train_loss)

        # Adjust metrics to get average loss and accuracy per batch
        train_loss = train_loss / len(dataloader)
        return train_loss

    def eval_epoch(model: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.Module):
        # Put model in eval mode
        model.eval()

        # Setup test loss and test accuracy values
        test_loss = 0

        # Turn on inference context manager
        with torch.no_grad():
            # Loop through DataLoader batches
            for batch, (X, y) in enumerate(dataloader):
                # Send data to target device
                X, y = X.to(device), y.to(device)

                # 1. Forward pass
                test_pred_logits = model(X)

                # 2. Calculate and accumulate loss
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.item()

        # Adjust metrics to get average loss and accuracy per batch
        test_loss = test_loss / len(dataloader)
        return test_loss

    loss_function = CalibrationLoss(0.1)

    dataset = CalibrationDataset(pref_vecs, queries, s)
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False, drop_last=True)

    model = CalibrationModel(in_features=2008, out_features=6, hidden_units=2048, num_heads=4, num_encoder_layers=6,
                             num_queries=s).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3 * 1e-4, betas=(0.9, 0.999), eps=1e-07, weight_decay=0)

    results = {"train_loss": [],
               "eval_loss": []
               }
    # Training for Calibration Model

    for epoch in tqdm(range(1, num_epochs + 1)):
        print("training for epoch:{}".format(epoch))
        train_loss = train_epoch(model=model,
                                 dataloader=train_dataloader,
                                 loss_fn=loss_function,
                                 optimizer=optimizer)

        print("evaluating for epoch: {}".format(epoch))
        eval_loss = eval_epoch(model=model,
                               dataloader=eval_dataloader,
                               loss_fn=loss_function)

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"eval_loss: {eval_loss:.4f} | "
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["eval_loss"].append(eval_loss)

        # 6. Return the filled results at the end of the epochs
    save_path = f'./{output_dir}/calibration_model.pth'
    torch.save(model.state_dict(), save_path)
    return results


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
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
                # train for gaze model and get the queries for training the calibration model
                queries, pref_vecs = train_gaze_model(group, output_dir, intial_num_epochs=40, num_epochs_cosine=40,
                                                      person_id=person_id, k_folds=k_folds)
                print(queries.shape)
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
                                            train_batch_size=8, eval_batch_size=8, person_id="p00", s=16)
                subject_id += 1





