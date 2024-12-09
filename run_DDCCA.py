
import torch
from DCCAmodel import DeepCCA
from DietDCCAmodel import DDCCA

try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle

import numpy as np
import pandas as pd
import random

torch.set_default_tensor_type(torch.DoubleTensor)

from sklearn.model_selection import KFold
import argparse
import os
import sys
import optuna
from solver import Solver_opt

import glob
import math

RNG_SEED = 0
# torch.manual_seed(RNG_SEED)
# random.seed(RNG_SEED)
# np.random.seed(RNG_SEED)


def objective(trial, tr_val1, tr_val2, rng_seed, model_name, model_filename, inner_K=4, b_size=200, n_out_size=50):
    """objective function for inner_cv

    Args:
        trial : optuna trial object
        tr_val1: train and validation set of df1
        tr_val2: train and validation set of df2
        rng_seed: random seed for reproducibility
        model_name: name of the model to use
        model_filename: filename of the best model among all folds and all trials (w/o .model)
        inner_K: number of folds for inner_cv

    Returns:
        mean of validation loss
    """

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise RuntimeError(f'The script {sys.argv[0]} needs GPU to run!')
    print("Using", torch.cuda.device_count(), "GPUs")

    save_to = 'EXPERIMENT/new_features.gz'

    params = {
        'lr': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'optimizer': 'Adam',
        'dr1': trial.suggest_float('dropout_rate1', 0.0, 0.95, step=0.05),
        'dr2': trial.suggest_float('dropout_rate2', 0.0, 0.95, step=0.05),
        'width': trial.suggest_int('n_units', 50, 200, step=10),
        'reg_par': trial.suggest_float('regularization_coeff', 1e-8, 1e-4, log=True)
    }

    print(f'Trial {trial.number} starts, with hyperparams: {trial.params}')

    epoch_num = 800
    batch_size = b_size

    if n_out_size == 0:
        n_out_size = math.floor(batch_size * (1.0 - max(params['dr1'], params['dr2']))) - 1
    nn_out_size = n_out_size
    outdim_size = 1

    input_shape1 = tr_val1.shape[1]
    input_shape2 = tr_val2.shape[1]

    layer_sizes1 = [params['width'], params['width'], params['width'], nn_out_size]
    if model_name == 'DDCCA':
        layer_sizes2 = {'emb_n_hidden_u': params['width'], 'discrim_n_hidden1_u': params['width'],
                        'discrim_n_hidden2_u': params['width']}
    elif model_name == 'DCCA':
        layer_sizes2 = [params['width'], params['width'], params['width'], nn_out_size]

    use_all_singular_values = False
    apply_linear_cca = True

    kf = KFold(n_splits=inner_K, shuffle=True, random_state=rng_seed)
    val_loss = []
    for i, (tr_idx, val_idx) in enumerate(kf.split(tr_val1)):
        tr_arr1, val_arr1 = tr_val1.iloc[tr_idx].values, tr_val1.iloc[val_idx].values
        tr_arr2, val_arr2 = tr_val2.iloc[tr_idx].values, tr_val2.iloc[val_idx].values

        tr1 = torch.tensor(tr_arr1)
        val1 = torch.tensor(val_arr1)
        tr2 = torch.tensor(tr_arr2).double()
        val2 = torch.tensor(val_arr2).double()

        # get emb(x.T) of genetics data
        if model_name == 'DDCCA':
            emb = tr_arr2.T
            emb = torch.from_numpy(emb)
            emb = emb.to(device)
            emb_norm = (emb ** 2).sum(0) ** 0.5
            emb = emb / emb_norm
            n_feats_emb = emb.size()[1]
        elif model_name == 'DCCA':
            emb = None

        if model_name == 'DDCCA':
            model = DDCCA(layer_sizes1, layer_sizes2, input_shape1, input_shape2,
                          n_feats_emb, nn_out_size, outdim_size, use_all_singular_values, device=device,
                          dropout_rate1=params['dr1'], dropout_rate2=params['dr2']).double()
        elif model_name == 'DCCA':
            model = DeepCCA(layer_sizes1, layer_sizes2, input_shape1, input_shape2,
                            outdim_size, use_all_singular_values, device=device,
                            dropout_rate1=params['dr1'], dropout_rate2=params['dr2']).double()
        l_cca = None
        if apply_linear_cca:
            l_cca = linear_cca()

        solver_opt = Solver_opt(trial, model, l_cca, outdim_size, epoch_num, batch_size,
                        params['lr'], params['reg_par'], device=device, optimizer_name=params['optimizer'])

        tmp_checkpt = f'{model_filename}-fold{i}.model'

        val_loss.append(solver_opt.fit(tr1, tr2, val1, val2, checkpoint=f'_{tmp_checkpt}', emb=emb))

    # find the best fold model file
    best_fold = np.array(val_loss).argmin()
    tmp_checkpt = f'{model_filename}-fold{best_fold}.model'

    trial.set_user_attr(key='best_fold_model_file', value=f'_{tmp_checkpt}')

    return np.median(val_loss)


def test_objective(best_trial_tr_val: optuna.trial.FrozenTrial, tr_val1, tr_val2, test1, test2, model_name, checkpt, b_size=200, n_out_size=50):
    '''
    Objective function to run the best model with the best hyperparams on test set, using frozen trial
    Shares most of the objective() function above with subtle changes

    '''
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise RuntimeError(f'The script {sys.argv[0]} needs GPU to run!')
    print("Using", torch.cuda.device_count(), "GPUs, when test objective")

    params = {
        'lr': best_trial_tr_val.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'optimizer': 'Adam',
        'dr1': best_trial_tr_val.suggest_float('dropout_rate1', 0.0, 0.95, step=0.05),
        'dr2': best_trial_tr_val.suggest_float('dropout_rate2', 0.0, 0.95, step=0.05),
        'width': best_trial_tr_val.suggest_int('n_units', 50, 200, step=10),
        'reg_par': best_trial_tr_val.suggest_float('regularization_coeff', 1e-8, 1e-4, log=True)
    }
    epoch_num = 800
    batch_size = b_size

    if n_out_size == 0:
        n_out_size = math.floor(batch_size * (1.0 - max(params['dr1'], params['dr2']))) - 1
    nn_out_size = n_out_size
    outdim_size = 1

    input_shape1 = test1.shape[1]
    input_shape2 = test2.shape[1]

    layer_sizes1 = [params['width'], params['width'], params['width'], nn_out_size]
    if model_name == 'DDCCA':
        layer_sizes2 = {'emb_n_hidden_u': params['width'], 'discrim_n_hidden1_u': params['width'],
                        'discrim_n_hidden2_u': params['width']}
    elif model_name == 'DCCA':
        layer_sizes2 = [params['width'], params['width'], params['width'], nn_out_size]

    use_all_singular_values = False
    apply_linear_cca = True

    checkpoint_ = torch.load(checkpt)
    emb = checkpoint_['emb']
    if model_name == 'DDCCA':
        # I. embed
        emb = emb.to(device)
        n_feats_emb = emb.size()[1]
        # II. build DDCCA model
        model = DDCCA(layer_sizes1, layer_sizes2, input_shape1, input_shape2,
                      n_feats_emb, nn_out_size, outdim_size, use_all_singular_values, device=device,
                      dropout_rate1=params['dr1'], dropout_rate2=params['dr2']).double()
    elif model_name == 'DCCA':
        model = DeepCCA(layer_sizes1, layer_sizes2, input_shape1, input_shape2,
                        outdim_size, use_all_singular_values, device=device,
                        dropout_rate1=params['dr1'], dropout_rate2=params['dr2']).double()

    l_cca = None
    if apply_linear_cca:
        l_cca = linear_cca()

    solver_opt = Solver_opt(best_trial_tr_val, model, l_cca, outdim_size, epoch_num, batch_size,
                            params['lr'], params['reg_par'], device=device,
                            optimizer_name=params['optimizer'])

    solver_opt.load_from_checkpt_dict(checkpoint_)

    test1 = torch.tensor(test1.values)
    test2 = torch.tensor(test2.values).double()
    loss = solver_opt.test(test1, test2, emb=emb)

    print('loss on test data: {:.4f}'.format(loss))

    return loss


def inner_cv(tr_val1, test1, tr_val2, test2, rng_seed, model_name, optuna_num_trials, inner_K=4, batch_size=200, nn_out_size=50):
    '''
    inner kfold cv, according to the best practice suggested by sklearn for nested cross-validation https://shorturl.at/gDJS9

    tr_val1, test1, tr_val2, test2: all pd dataframes
    '''

    sampler = optuna.samplers.TPESampler(seed=rng_seed, multivariate=True) # other samplers
    study_name = f'{model_name}-seed{rng_seed}_{os.getpid()}'
    study = optuna.create_study(
        sampler=sampler,
        study_name=study_name,
        direction="minimize")

    model_filename = study_name
    best_trial_model_file = f'{model_filename}.model'
    study.optimize(
        lambda trial_: objective(trial_, tr_val1, tr_val2, rng_seed, model_name, model_filename, inner_K,
                                 batch_size, nn_out_size),
        callbacks=[lambda study_, trial_: callback(study_, trial_, model_filename)],
        n_trials=optuna_num_trials)

    for fold_fn in glob.glob(f'_{model_filename}-fold*'):
        os.remove(fold_fn)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    best_trial_tr_val = study.best_trial
    print("Best trial:")
    print("  Value: ", best_trial_tr_val.value)
    print("  Params: ")
    for key, value in best_trial_tr_val.params.items():
        print("    {}: {}".format(key, value))

    #load best model to do test
    loss_test = test_objective(best_trial_tr_val, tr_val1, tr_val2, test1, test2, model_name, best_trial_model_file,
                               b_size=test1.shape[0]+1,
                               n_out_size=nn_out_size)

    return loss_test


def callback(study, frozen_trial, model_fn):
    '''
    rename the best fold model file to the model_fn. remove all the other folds model files
    '''
    print(f'Trial {frozen_trial.number} finished, obtaining objective={frozen_trial.value}')

    if study.best_trial.number == frozen_trial.number:
        os.replace(frozen_trial.user_attrs['best_fold_model_file'], f'{model_fn}.model')

        print(f'By far, the best trial is {frozen_trial.number}, obtaining objective={frozen_trial.value}, hyperparams: {frozen_trial.params}')


def kFoldCV_optuna(df1, df2, rng_seed, model_name, optuna_num_trials, K=5, batch_size=200, nn_out_size=50, nested=False):
    '''
    nested: sequential nested CV (removed, and instead do parallel nested CV on cluster to accelerate)

    this function is just the outer_cv, no inner_cv
    '''
    if nested:
        pass   # removed to do parallel instead
    else:
        idx = np.arange(len(df1))  # df1 and df2 should have the same length
        np.random.seed(rng_seed)
        np.random.shuffle(idx)
        test_idx, tr_val_idx = np.split(idx, [int((1 / K) * len(df1))])  # randomly held out a test set
        test1, tr_val1 = df1.iloc[test_idx], df1.iloc[tr_val_idx]
        test2, tr_val2 = df2.iloc[test_idx], df2.iloc[tr_val_idx]
        loss_test = inner_cv(tr_val1, test1, tr_val2, test2, rng_seed, model_name, optuna_num_trials, K-1,
                             batch_size, nn_out_size)

        return loss_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run optuna experiments')
    parser.add_argument('--in_imgfile', type=str,
                        help='input imaging filename with full path',
                        default='DATA/image.csv')
    parser.add_argument('--in_snpfile', type=str,
                        help='input genotype filename with full path',
                        default='DATA/genetic.csv')
    parser.add_argument('--rng_seed', type=int,
                        help='random number generator seed; important to fix to be the same to compare DCCA and DDCCA '
                             'on the same kfold splits',
                        default=RNG_SEED)
    parser.add_argument('--which_model', choices=['DCCA', 'DDCCA'],
                        help='choose from DCCA or DDCCA model',
                        required=True)
    parser.add_argument('--optuna_num_trials', type=int,
                        help='number of trials for optuna experiment',
                        default=100)
    parser.add_argument('--batch_size', type=int,
                        help='batch size for training',
                        default=200)
    parser.add_argument('--nn_out_size', type=int,
                        help='nn output size for computing corr',
                        default=50)
    parser.add_argument('--num_folds', type=int,
                        help='number of folds for k-fold CV',
                        default=5)

    args = parser.parse_args()

    # first fix all random number generator seeds for reproducibility
    torch.manual_seed(args.rng_seed)
    random.seed(args.rng_seed)
    np.random.seed(args.rng_seed)

    # detect nan gradient
    torch.autograd.set_detect_anomaly(True)

    # prepare k-fold data
    df1 = pd.read_csv(args.in_imgfile)
    df2 = pd.read_csv(args.in_snpfile)

    K = args.num_folds

    neg_corr = kFoldCV_optuna(df1, df2, args.rng_seed, args.which_model, args.optuna_num_trials, K=K,
                              batch_size=args.batch_size, nn_out_size=args.nn_out_size)

    print(f"seed {args.rng_seed}, {args.which_model} neg_corr on test data: {neg_corr}")

