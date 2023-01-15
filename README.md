# Sequence-tagging using Deep Learning
## Requirements
* PyTorch version >= 1.13.1+cu117
* Python version >= 3.10.4
* PyTorch-Lightning version >= 1.8.6
* Huggingface Transformers version >= 4.25.1
* Tensorboard version >= 2.11.0
* Pandas >= 1.5.2
* Scikit-learn >= 1.2.0
* Seqeval >= 1.2.2
## Installation
```
pip install transformers
pip install pytorch-lightning
pip install tensorboard
pip install pandas
pip install scikit-learn
pip install seqeval
git clone https://github.com/vineetk1/sequence-tagging.git
cd sequence-tagging
```
Note that the default directory is *sequence-tagging*. Unless otherwise stated, all commands from the Command-Line-Interface must be delivered from the default directory.
## Download the dataset
1. Create a *data* directory.      
```
mkdir data
```
2. Download a dataset in the *data* directory.       
## Saving all informtion and results of an experiment
All information about the experiment is stored in a unique directory whose path starts with tensorboard_logs and ends with a unique version-number. Its contents consist of hparams.yaml, hyperperameters_used.yaml, test-results.txt, dialogs_failed.txt, events.* files, and a checkpoints directory that has one or more checkpoint files.
## Train, validate, and test a model
Following command trains a model, saves the last checkpoint plus checkpoints that have the lowest validation loss, runs the test dataset on the checkpointed model with the lowest validation loss, and outputs the results of the test:
```
python3 Main.py input_param_files/bert_seq_tag
```
The user-settable hyper-parameters are in the file *input_param_files/bert_seq_tag*. An explanation on the contents of this file is at *input_param_files/README.md*. A list of all the hyper-parameters is in the <a href="https://www.pytorchlightning.ai" target="_blank">PyTorch-Lightning documentation</a>, and any hyper-parameter can be used.    
To assist in Training, the two parameters *auto_lr_find* and *auto_scale_batch_size* in the file *input_param_files/bert_seq_tag* enable the software to automatically find an initial Learning-Rate and a Batch-Size respectively.    
As training progresses, graphs of *"training-loss vs. epoch #"*, *"validation-loss vs. epoch #"*, and "learning-rate vs. batch #" are plotted in real-time on the TensorBoard.  Training is stopped by typing, at the Command-Line-Interface, the keystroke ctrl-c. The current training information is checkpointed, and training stops. Training can be resumed, at some future time, from the checkpointed file.   
During testing, the results are sent to the standard-output, and also saved in the *test-results.txt" file that include the following: general information about the dataset and the tags, confusion matrix, precision, recall, f1, average f1, and weighted f1.
## Resume training, validation, and testing a model with same hyper-parameters
Resume training a checkpoint model with the same model- and training-states by using the following command:
```
python3 Main.py input_param_files/bert_seq_tag-resume_from_chkpt
```
The user-settable hyper-parameters are in the file *input_param_files/bert_seq_tag-res_from_chkpt*.  An explanation on the contents of this file is at *input_param_files/README.md*.
## Change hyper-parameters and continue training, validation, and testing a model
Continue training a checkpoint model with the same model-state but different hyperparameters for the training-state by using the following command:
```
python3 Main.py input_param_files/bert_seq_tag-ld_chkpt
```
The user-settable hyper-parameters are in the file *input_param_files/bert_seq_tag-ld_chkpt*.  An explanation on the contents of this file is at *input_param_files/README.md*.   
## Further test a checkpoint model with a new dataset
Test a checkpoint model by using the following command:
```
python3 Main.py input_param_files/bert_seq_tag-ld_chkpt_and_test
```
The user-settable hyper-parameters are in the file *input_param_files/bert_seq_tag-ld_chkpt_and_test*.  An explanation on the contents of this file is at *input_param_files/README.md*.
