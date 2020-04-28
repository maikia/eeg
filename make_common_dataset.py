import glob
import numpy as np
import pandas as pd
import os

# read the data from the given subjects
subjects = 'all'
signal_type = 'grad'
n_parcels_max = 3
no_parcels = 26

# if False makes the combined data from scratch, otherwise adds only those
# subjects which are not yet in the all data

# combine them in one datafile with each sample: data from electrodes,
# subject_name, signal_type
data_type = '_' + str(no_parcels) + '_' + str(n_parcels_max)
data_dir_all = 'data/data_' + signal_type + '_all' + data_type
data_dir = ('data/data_' + signal_type + '_*' + data_type)
data_dirs = sorted(glob.glob(data_dir))

if not os.path.isdir(data_dir_all):
    os.mkdir(data_dir_all)
else:
    pass
    # TODO: empty existing path

# initialize the files in the data_dir_all

all_X_file = os.path.join(data_dir_all, 'X.csv')
# TODO: remove data_grad_all from the data dirs
for idx, subject_path in enumerate(data_dirs):
    subject_info = subject_path.split('_')
    print('adding subject ' + subject_info[2])
    subject_data = pd.read_csv(os.path.join(subject_path, 'X.csv'))
    subject_data['subject'] = subject_info[2]

    target_data = np.load(os.path.join(subject_path, 'target.npz'))
    import pdb; pdb.set_trace()

    try:
        # append the data
        subject_data.to_csv(all_X_file, mode='a', header=False, index=False)
    except:
        # create new .csv file
        subject_data.to_csv(all_X_file, header=True, index=False)

    # import pdb; pdb.set_trace()
    # read the data

    # df.to_csv(os.path.join(data_dir_specific, 'X.csv'), index=False)
    # save_npz(os.path.join(data_dir_specific, 'target.npz'), target)
    # print(str(len(df)), ' samples were saved')
#    pass
# if not combine:
    # make new data folder
# read the subject data one by one and