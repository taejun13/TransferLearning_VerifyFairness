from json import dumps
import sys
import os
import tensorflow as tf
from codebase.datasets import Dataset_cfair
from codebase import models
from codebase.trainer import Trainer
from codebase.tester import Tester
from codebase.results import ResultLogger
from codebase.utils import get_npz_basename, make_dir_if_not_exist
from codebase.metrics import *

# modified by whie
import model_cfair
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
import torch
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
from tensorboard_logger import configure, log_value

def main(args):
    # things with paths
    expdname = args['dirs']['exp_dir']
    expname = args['exp_name']
    logdname = args['dirs']['log_dir']
    resdirname = os.path.join(expdname, expname)
    #logdirname = os.path.join(logdname, expname)
    logdirname = resdirname
    make_dir_if_not_exist(logdirname, remove=True)
    make_dir_if_not_exist(resdirname)
    npzfile = os.path.join(args['dirs']['data_dir'],
                           args['data']['name'],
                           get_npz_basename(**args['data']))

    # write params (including all overrides) to experiment directory
    with open(os.path.join(resdirname, 'opt.json'), 'w') as f:
        opt_dumps = dumps(args, indent=4, sort_keys=True)
        f.write(opt_dumps)

    if args['data']['use_attr']:
        args['model'].update(xdim=args['model']['xdim']+1)


    #get dataset
    data = Dataset_cfair(npzfile=npzfile, **args['data'], batch_size=args['train']['batch_size'])

    # get model
    if 'Weighted' in args['model']['class']:
        A_weights = [1. / x for x in data.get_A_proportions()]
        Y_weights = [1. / x for x in data.get_Y_proportions()]
        AY_weights = [[1. / x for x in L] for L in data.get_AY_proportions()]
        if 'Eqopp' in args['model']['class']:
            #we only care about ppl with Y = 0 --- those who didn't get sick
            AY_weights[0][1] = 0. #AY_weights[0][1]
            AY_weights[1][1] = 0. #AY_weights[1][1]
        args['model'].update(A_weights=A_weights, Y_weights=Y_weights, AY_weights=AY_weights)

    # model_class = getattr(models, args['model'].pop('class'))
    # model = model_class(**args['model'], batch_size=args['train']['batch_size'])

    #methods = ['nodebias', 'laftr','cfair_EO','cfair']
    #methods = ['nodebias','laftr', 'cfair', 'cfair_EO','cfair_ours']
    methods = ['cfair', 'cfair_EO']
    trials = 10
    results = {}
    summary = {}
    for n in range(trials):
        for method in methods:
            if n==0:
                results[method] = {}

            performance = train_nets(args, data, method, lamb=1000.)
            for key in performance.keys():
                if key not in results[method].keys():
                    results[method][key] = []
                else:
                    results[method][key].append(performance[key])
    
    for method in methods:
        summary[method]={}
        for key in results[method].keys():
            tmp_perform = results[method][key]
            tmp_perform = np.array(tmp_perform)

            summary[method][key] = [tmp_perform.mean(), tmp_perform.std()]

    print(summary)

def ber(output, target, y_NR, y_PR):
    # balanced error rate relaxed with CE Loss.
    eps = 1e-10
    sig = nn.Sigmoid()
    sig_out = sig(output)
    _ber = -target * torch.log(sig_out + eps) / y_PR - (1-target) * torch.log(1-sig_out + eps) / y_NR
    return _ber * 0.5


def train_nets(args, data, method, lamb):

    # define networks
    net_classifier = model_cfair.Classifier(include_A=args['data']['use_attr']).cuda()
    net_adversary1 = model_cfair.Adversary().cuda()
    net_adversary2 = model_cfair.Adversary().cuda()

    # define optimizers
    optim_cls = optim.Adadelta(net_classifier.parameters(), lr=1.0) 
    optim_adv = optim.Adadelta(list(net_adversary1.parameters()) + list(net_adversary2.parameters()), lr=1.0) 
    
    # criterion
    ce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()
    sig = nn.Sigmoid()

    # global iteration 
    global_iter = 0

    # writer = SummaryWriter('tf_logs_cfair')
    # configure(f'tf_logs_cfair/{method}', flush_secs = 10)

    y_NR, y_PR = data.get_Y_proportions()
    A_NR, A_PR = data.get_A_proportions()
    ya_proportion = data.get_YA_proportions()
    Y_NR_A_NR, Y_NR_A_PR = ya_proportion[0]
    Y_PR_A_NR, Y_PR_A_PR = ya_proportion[1]

    
    for epoch in range(args['train']['n_epochs']): 
        # start training
        train_iter = data.get_batch_iterator('train', args['train']['batch_size'])
        
        count=0

        for x, y, a in train_iter:
            x = x.cuda()
            y = y.cuda()
            a = a.cuda()

            # max h', h''
            # encode into hidden vectors
            h_vec, _ = net_classifier(x)

            if method is 'nodebias':
                if epoch > 20:
                    break

                h_vec, pred_y = net_classifier(x)
                loss_cls = ce(pred_y, y).mean()

                optim_cls.zero_grad()
                loss_cls.backward()
                optim_cls.step()

                # if global_iter % 100:
                #     log_value('L_cls', loss_cls, global_iter)
 
            else:
                # adversary
                pred_a_y1 = net_adversary1(h_vec)
                pred_a_y2 = net_adversary2(h_vec)


                mask1 = torch.zeros_like(y).cuda().detach()
                mask1[y==0] = 1 
                mask2 = 1 - mask1
               
                if method is 'cfair':
                    loss_adv = mask1 * ber(pred_a_y1, a, Y_NR_A_NR, Y_NR_A_PR) + mask2 * ber(pred_a_y2, a, Y_PR_A_NR, Y_PR_A_PR)
                    # loss_adv = mask1 * ber(pred_a_y1, a, A_NR, A_PR) + mask2 * ber(pred_a_y2, a, A_NR, A_PR)
                    # loss_adv = mask1 * cespred_a_y1, a) + mask2 * ce(pred_a_y2, a)
                elif method is 'cfair_ours':
                    loss_adv = mask1 * ce(pred_a_y1, a) + mask2 * ce(pred_a_y2, a)

                else:
                    loss_adv = ce(pred_a_y1, a)

                loss_adv = lamb * loss_adv.mean()

                optim_adv.zero_grad()
                loss_adv.backward()
                optim_adv.step()

                for i in range(2):
                    # min h, g
                    # feed forward
                    h_vec, pred_y = net_classifier(x)
                    
                    # adversary
                    pred_a_y1 = net_adversary1(h_vec)
                    pred_a_y2 = net_adversary2(h_vec)

                    if method is 'cfair':
                        loss_y = ber(pred_y, y, y_NR, y_PR)
                        loss_adv_cls = mask1 * ber(pred_a_y1, a, Y_NR_A_NR, Y_NR_A_PR) + mask2 * ber(pred_a_y2, a, Y_PR_A_NR, Y_PR_A_PR)
                        # loss_adv_cls = mask1 * ber(pred_a_y1, a, A_NR, A_PR) + mask2 * ber(pred_a_y2, a, A_NR, A_PR)
                        # loss_adv_cls = (mask1 * ce(pred_a_y1, a) + mask2 * ce(pred_a_y2, a)).mean()
                    
                    elif method is 'cfair_ours':
                        loss_y = ber(pred_y, y, y_NR, y_PR)
                        loss_adv_cls = mask1 * ce(pred_a_y1, a) + mask2 * ce(pred_a_y2, a)

                    elif method is 'cfair_EO':
                        loss_y = ce(pred_y, y)
                        loss_adv_cls = mask1 * ber(pred_a_y1, a, Y_NR_A_NR, Y_NR_A_PR) + mask2 * ber(pred_a_y2, a, Y_PR_A_NR, Y_PR_A_PR)

                    elif method is 'laftr':
                        loss_y = ce(pred_y, y)
                        loss_adv_cls = l1(sig(pred_a_y1), a)
                    
                    if loss_y.mean() != loss_y.mean():
                        import ipdb; ipdb.set_trace(context=15)
                        a = 1
                    loss_y = loss_y.mean()
                    loss_adv_cls = loss_adv_cls.mean()
                    loss_cls = loss_y - lamb * loss_adv_cls

                    optim_cls.zero_grad()
                    loss_cls.backward()
                    optim_cls.step()

                # if global_iter % 300:
                #     log_value('L_adv', loss_adv, global_iter)
                #     log_value('L_adv_cls', loss_adv_cls, global_iter)
                #     log_value('L_cls', loss_y, global_iter)
         
                    # print(f"{global_iter} || loss_adv = {loss_adv.item():.5f} loss_adv_cls = {loss_adv_cls.item():.5f} || loss_y = {loss_y.item():.5f}")

            count += 1
            global_iter += 1

        # start test
        test_iter = data.get_batch_iterator('test', args['train']['batch_size'])
        test_L = {'class': 0., 'disc': 0., 'class_err': 0., 'disc_err': 0., 'recon': 0}
        num_batches = 0
        Y_hats = np.empty((0, 1))
        A_hats = np.empty((0, 1))
        Ys = np.empty((0, 1))
        As = np.empty((0, 1))

        for x, y, a in test_iter:
            with torch.no_grad():
                x = x.cuda()
                y = y.cuda()
                a = a.cuda()

                # max h', h''
                # encode into hidden vectors
                h_vec, pred_y = net_classifier(x)
                # adversary
                pred_a_y1 = net_adversary1(h_vec)
                pred_a_y2 = net_adversary2(h_vec)

                out_y = torch.round(nn.Sigmoid()(pred_y))
                
                Y_hats = np.concatenate((Y_hats, out_y.cpu().numpy()))
                Ys = np.concatenate((Ys, y.cpu().numpy()))
                As = np.concatenate((As, a.cpu().numpy()))

        # err1, err2 w.r.t A. 
        Err1 = subgroup(errRate, As, Ys, Y_hats)
        Err2 = subgroup(errRate, 1-As, Ys, Y_hats)
        ErrGap = abs(Err1-Err2)
        JointErr = Err1+Err2

        # Equalized Odd Gap.
        difp = DI_FP(Ys, Y_hats, As)
        difn = DI_FN(Ys, Y_hats, As)

        #EO = difp + difn
        EO = max(difp, difn)
        
        DemoP = DP(Y_hats, As)
        # print(f'GT DP = {DP(Ys, As)}')

        # import ipdb; ipdb.set_trace(context=15)
        print(f'METHOD={method}, LAMBDA={lamb} || EPOCH={epoch} ErrGAP : {ErrGap:.3f} | JointErr : {JointErr:.3f} | EO : {EO:.3f} | DP : {DemoP:.3f}')

    return {'ErrGap':ErrGap, 'JointErr':JointErr, 'EO':EO, 'DemoP':DemoP,'Lamb':lamb}



if __name__ == '__main__':
    """
    This script trains a LAFTR model. For the full evaluation used in the paper (first train LAFTR then evaluate on naive classifier) see `src/run_laftr.py`.

    Instructions: 
    1) Run from repo root
    2) First arg is base config file
    3) Optionally override individual params with -o then comma-separated (no spaces) list of args, e.g., -o exp_name=foo,data.seed=0,model.fair_coeff=2.0 
       (the overrides must come before the named templates in steps 4--5)
    4) Set templates by name, e.g., --data adult 
    5) Required templates are --data and --dirs
    
    e.g.,
    >>> python src/laftr.py conf/laftr/config.json -o train.n_epochs=10,model.fair_coeff=2. --data adult --dirs local

    This command trains LAFTR on the Adult dataset for ten epochs with batch size 32.
    Model and optimization parameters are specified by the config file conf/classification/config.json
    Dataset specifications are read from conf/templates/data/adult.json
    Directory specifications are read from conf/templates/dirs/local.json
    Finally, two hyperparameters are overridden by the last command.
    By using the -o flag we train for 10 epochs with fairness regulariztion coeff 2. instead of the default values from the config.json.
    """
    from codebase.config import process_config
    opt = process_config(verbose=False)
    main(opt)

