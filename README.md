# pgte_eye_gaze
Code adaptation of the PGTE paper([link](http://jcse.kiise.org/files/V17N2-01.pdf)) for eye gaze prediction.

# Setting up datasets and preprocessing
Download the datasets and note down their absolute or relative paths: [MPIIFaceGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation/) [GazeCapture](https://gazecapture.csail.mit.edu/download.php)

Run the below commands to set up python environment:

*bash grab_prerequisites.bash*

the bash file will download supplementary h5 dataset files for the datasets that contain face features in 3d and face rotation matrix.

For preprocessing run the "create_hdf_files_for_pgte.py" file. This will create a new h5 dataset that is compatible with our model. This will contain necessary: face, right eye, flipped face and flipped eye images and other features mentioned in the paper. Kindly don't forget to modify dataset paths in the file before running.

*python create_hdf_files_for_pgte.py*


For running the training code, run the train_eval.py file. This will train a gaze model and calibration model per person or subject. Kindly look in the file main code to change the hyper parameters.

*python train_eval.py*
