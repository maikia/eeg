import glob
import os
import pandas as pd
import random
from scipy import sparse
from scipy.sparse import save_npz
import shutil


# read the data from the given subjects
subjects = 'all'
signal_type = 'grad'
n_parcels_max = 3
no_parcels = 450
data_dir_base = 'data/train/'
# assign a number from which first subject_id will start
if 'train' in data_dir_base:
    subject_id_start = 1
    subdir = 'train'
elif 'test' in data_dir_base:
    subject_id_start = 6
    subdir = 'test'
divide_to = 2  # if 1 data will not be split. Otherwise it might be 2 or 4.
# the data will be equally split to private and public (if set 2) or
# to private/public/test/train (if set to 4)
data_dir_save = 'data/ramp_challenge'

assert divide_to == 1 or divide_to == 2 or divide_to == 4

# combine them in one datafile with each sample: data from electrodes,
# subject_name, signal_type
data_type = '_' + str(no_parcels) + '_' + str(n_parcels_max)

data_dir_all = os.path.join(data_dir_base,
                            f'data_{signal_type}_all{data_type}')
data_dir = os.path.join(data_dir_base, f'data_{signal_type}_*{data_type}')
if data_dir_save is None:
    data_dir_save = data_dir_all

data_dirs = sorted(glob.glob(data_dir))
# if the all_data directory is still in the list of dirs, remove it
if data_dir_all in data_dirs:
    data_dirs.remove(data_dir_all)
assert data_dir_all not in data_dirs
random.seed(42)
random.shuffle(data_dirs)


def save_the_data(data_dir_all, subdir, data_dirs, subj_id_init,
                  remove_previous=False):
    # data_dir_all : where to save the data
    # data_dirs : paths of the saved data
    # remove_previous : clean up the data_dir_all directory before saving new
    #   files
    sbj_id = 0
    # initialize the files in the data_dir_all
    data_dir_all_subdir = os.path.join(data_dir_all, subdir)
    all_X_file = os.path.join(data_dir_all_subdir, 'X.csv.gz')

    # remove previous save directory and create a clean one
    if os.path.isdir(data_dir_all) and remove_previous:
        print('removing contents from ' + data_dir_all)
        shutil.rmtree(data_dir_all)
    if not os.path.exists(data_dir_all):
        os.makedirs(data_dir_all)

    if not os.path.exists(data_dir_all_subdir):
        os.makedirs(data_dir_all_subdir)

    for idx, subject_path in enumerate(data_dirs):
        # check if all the necessary files are present
        subject_info = subject_path.split('_')
        subject_name = subject_info[-3]
        subject_name_id = 'subject_' + str(subj_id_init + idx)

        # make sure that all the necessary files are in the directory
        labels_exist = os.path.exists(os.path.join(subject_path,
                                      subject_name + '_labels.npz'))
        target_exists = os.path.exists(os.path.join(subject_path,
                                                    'target.npz'))
        X_exists = os.path.exists(os.path.join(subject_path, 'X.csv'))
        lf_exists = os.path.exists(os.path.join(subject_path,
                                                'lead_field.npz'))

        if not (labels_exist and target_exists and X_exists and lf_exists):
            print(f'skipping {subject_path}.'
                  'not all the necessary files are present')
            continue

        print(f'adding subject {subject_name} as {subject_name_id}')
        subject_data = pd.read_csv(os.path.join(subject_path, 'X.csv'))
        subject_data['subject'] = subject_name_id
        target_subject = sparse.load_npz(os.path.join(subject_path,
                                                      'target.npz'))

        if sbj_id == 0:
            # create new .csv file
            subject_data.to_csv(all_X_file, header=True, index=False,
                                compression='gzip')
            target_all = target_subject
        else:
            # append the data
            subject_data.to_csv(all_X_file, mode='a', header=False,
                                index=False, compression='gzip')
            target_all = sparse.vstack((target_all, target_subject))
        # shutil.copyfile(os.path.join(subject_path, 'labels.pickle'),
        #               os.path.join(data_dir_all,
        #                            subject_name + '_labels.pickle')
        #                )

        shutil.copyfile(os.path.join(subject_path, 'lead_field.npz'),
                        os.path.join(data_dir_all,
                                     subject_name_id + '_L.npz')
                        )
        # uncomment if you want to also save labels
        # shutil.copyfile(os.path.join(subject_path,
        #                              subject_name + '_labels.npz'),
        #                 os.path.join(data_dir_all,
        #                              subject_name_id + '_labels.npz')
        #                 )
        sbj_id += 1
    if len(data_dirs):
        # save the target
        save_npz(os.path.join(data_dir_all_subdir, 'target.npz'), target_all)
        print(f'{target_all.shape[0]} samples from {sbj_id} subjects'
              f' were saved in the {data_dir_all} + {subdir}')
    else:
        print(f'Did not find any files in {data_dir}')


if len(data_dirs) % divide_to != 0:
    print('here we are using equal number of data files in each directory.'
          f'{len(data_dirs) % divide_to} of the files will not be used')
    data_dirs = data_dirs[:-len(data_dirs) % divide_to]
if divide_to == 4:
    # divide all the data to public/train, public/test, private/test and
    # private/train equally
    n_files = int(len(data_dirs) / 4)
    # public/train
    public_train_dir = os.path.join(data_dir_save, 'public')
    save_the_data(public_train_dir, subdir='train',
                  data_dirs=data_dirs[:n_files],
                  subj_id_init=subject_id_start)
    # public/test
    public_test_dir = os.path.join(data_dir_save, 'public')
    save_the_data(public_test_dir, subdir='test',
                  data_dirs=data_dirs[n_files:n_files*2],
                  subj_id_init=subject_id_start+n_files)
    # private/train
    private_train_dir = os.path.join(data_dir_save, 'private')
    save_the_data(private_train_dir, subdir='train',
                  data_dirs=data_dirs[n_files*2:n_files*3],
                  subj_id_init=subject_id_start+n_files*2)
    # private/test
    private_test_dir = os.path.join(data_dir_save, 'private')
    save_the_data(private_test_dir, subdir='test',
                  data_dirs=data_dirs[n_files*3:],
                  subj_id_init=subject_id_start+n_files*3)
elif divide_to == 2:
    # divide all the data to public and private equally
    n_files = int(len(data_dirs) / 2)
    # public/train
    public_train_dir = os.path.join(data_dir_save, 'public')
    save_the_data(public_train_dir, subdir,
                  data_dirs=data_dirs[:n_files],
                  subj_id_init=subject_id_start)
    # public/test
    public_test_dir = os.path.join(data_dir_save, 'private')
    save_the_data(public_test_dir, subdir,
                  data_dirs=data_dirs[n_files:],
                  subj_id_init=subject_id_start)


elif divide_to == 1:
    save_the_data(data_dir_save, data_dirs, subj_id_init=subject_id_start)
