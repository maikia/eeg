import glob
import os
import pandas as pd
from scipy import sparse
from scipy.sparse import save_npz
import shutil


# read the data from the given subjects
subjects = 'all'
signal_type = 'grad'
n_parcels_max = 3
no_parcels = 450
data_dir_base = 'data/challenge'
subject_id_start = 1  # assign a number from which first subject_id will start

# combine them in one datafile with each sample: data from electrodes,
# subject_name, signal_type
data_type = '_' + str(no_parcels) + '_' + str(n_parcels_max)

data_dir_all = os.path.join(data_dir_base,
                            f'data_{signal_type}_all{data_type}')
data_dir = os.path.join(data_dir_base, f'data_{signal_type}_*{data_type}')

if os.path.isdir(data_dir_all):
    print('removing contents from ' + data_dir_all)
    shutil.rmtree(data_dir_all)
data_dirs = sorted(glob.glob(data_dir))
os.mkdir(data_dir_all)

# if the all_data directory is still in the list of dirs, remove it
if data_dir_all in data_dirs:
    data_dirs.remove(data_dir_all)
assert data_dir_all not in data_dirs

sbj_id = 0
# initialize the files in the data_dir_all
all_X_file = os.path.join(data_dir_all, 'X.csv.gz')
for idx, subject_path in enumerate(data_dirs):
    # check if all the necessary files are present
    subject_info = subject_path.split('_')
    subject_name = subject_info[-3]
    subject_name_id = 'subject_' + str(subject_id_start + idx)

    labels_exist = os.path.exists(os.path.join(subject_path,
                                  subject_name + '_labels.npz'))
    target_exists = os.path.exists(os.path.join(subject_path, 'target.npz'))
    X_exists = os.path.exists(os.path.join(subject_path, 'X.csv'))
    lf_exists = os.path.exists(os.path.join(subject_path, 'lead_field.npz'))

    if not (labels_exist and target_exists and X_exists and lf_exists):
        print('skipping {}. not all the necessary files are present'.format(
              subject_path))
        continue

    print(f'adding subject {subject_name} as {subject_name_id}')
    subject_data = pd.read_csv(os.path.join(subject_path, 'X.csv'))
    subject_data['subject'] = subject_name_id
    target_subject = sparse.load_npz(os.path.join(subject_path, 'target.npz'))

    if sbj_id == 0:
        # create new .csv file
        subject_data.to_csv(all_X_file, header=True, index=False,
                            compression='gzip')
        target_all = target_subject
    else:
        # append the data
        subject_data.to_csv(all_X_file, mode='a', header=False, index=False,
                            compression='gzip')
        target_all = sparse.vstack((target_all, target_subject))
    # shutil.copyfile(os.path.join(subject_path, 'labels.pickle'),
    #               os.path.join(data_dir_all, subject_name + '_labels.pickle')
    #                )
    shutil.copyfile(os.path.join(subject_path, 'lead_field.npz'),
                    os.path.join(data_dir_all,
                                 subject_name_id + '_lead_field.npz')
                    )
    # uncomment if you want to also save labels
    # shutil.copyfile(os.path.join(subject_path, subject_name + '_labels.npz'),
    #                 os.path.join(data_dir_all,
    #                              subject_name_id + '_labels.npz')
    #                 )
    sbj_id += 1
if len(data_dirs):
    # save the target
    save_npz(os.path.join(data_dir_all, 'target.npz'), target_all)
    print('{} samples from {} subjects were saved in the {}'.format(
        target_all.shape[0], sbj_id, data_dir_all))
else:
    print(f'Did not find any files in {data_dir}')
