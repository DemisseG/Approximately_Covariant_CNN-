import optuna
import trainlogic as tl 
import argparse 
import config 
import os

import torch 

STUDY_CASE = 'rotmnist'

def objective(trial):
    # selected variables to explore 
    config.reset_report()
    lr = trial.suggest_float('lr', 0.05, 0.2) # use to be 0.001 to 0.1 for adam
    batch = trial.suggest_int('batch', 50, 120) # 50 - 120 is good for optimal value selection for resent 8
    conv_type = trial.suggest_categorical('conv_type', ['conv', 'covar'])

    # objective to be optimized
    tl.parse()
    config.PERFORMANCE_DETAILS = 0 
    config.best_acc = 0.0

    # hyper-parameters to search for
    config.args.batch = batch
    config.args.l_rate = lr
    config.args.convType = conv_type 
    config.args.symset = 4
    if STUDY_CASE == 'cifar':
        config.args.depth = 10 
        config.args.datasets =  'cifar10' 
        config.args.imode = 0 
        config.args.symset = 4
        config.args.optim = 'sgd' 
    elif STUDY_CASE == 'rotmnist':
        config.args.depth = 8
        config.args.trainType = 1
        config.args.datasets =  'rotmnist' 
        config.args.imode = 1 
        config.args.symset = 4
        config.args.optim = 'adam' 
     
    # model and data
    config.args.datapath = os.getcwd() + '/data'
    tl.prepare_data() 
    tl.verbose()
    tl.main()
    
    # objective is based on the average precision (@ top 1 cut off) score of the model over the last 10 epochs.
    res = torch.stack([e[0] for e in config.report['Progress']['test_acc'][-10:]])
    return res.mean().item()


def spwan_study(arg):
    study_name = arg['study_name']
    storage = "sqlite:///par_optim/{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage, direction='maximize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=40), 
                                load_if_exists=True)
    study.optimize(objective, n_trials=arg['num_trails']) 
    return study 

if __name__ == '__main__':
    arg = {}
    arg['study_name'] = 'study_conv_covar_ref-scale_cifar10_res10_3' #'test_study_exact' 
    arg['num_trails'] = 10
    study = spwan_study(arg)
    
    print("\n [ Optimal values summary ] ")
    print(" ============================")
    print(' | Best parameters ', study.best_params)
    print(' | Best result', study.best_value)