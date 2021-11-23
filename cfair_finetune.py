from json import dumps
import sys
import time
import os
# import tensorflow as tf
from datasets import HealthDataset
from src.codebase.metrics import *

# modified by whie
from argparse import ArgumentParser
import model_cfair
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from tensorboard_logger import configure, log_value

def main(args):
    #get dataset
    data = {}

    args.dataset = 'health'
    data['train'] = HealthDataset(stage='train')
    data['test'] = HealthDataset(stage='test')
    data['transfer_train'] = HealthDataset(stage='transfer_train')
    data['transfer_test'] = HealthDataset(stage='transfer_train')

    args.dim_input = 69
    args.dim_hid_cls = 20
    args.dim_hid_adv = 20

    train_loader = DataLoader(data['train'], batch_size=args.batch, shuffle=True) 
    test_loader = DataLoader(data['test'], batch_size=args.batch, shuffle=True) 
    tf_train_loader = DataLoader(data['transfer_train'], batch_size=args.batch, shuffle=True) 
    tf_test_loader = DataLoader(data['transfer_test'], batch_size=args.batch, shuffle=False) 
    loaders = {'train_loader':train_loader,'test_loader':test_loader, 'tf_train_loader':tf_train_loader,'tf_test_loader':tf_test_loader}

    # set data proportions
    proportions = {'train':{}, 'test':{}}
    modes = ['train','test']
    for mode in modes:
        A0, A1 = data[mode].get_A_proportions()
        Y0, Y1 = data[mode].get_Y_proportions()
        base_A0, base_A1 = data[mode].get_YA_proportions()
        proportions[mode]['A0'] = A0
        proportions[mode]['A1'] = A1
        proportions[mode]['Y0'] = Y0
        proportions[mode]['Y1'] = Y1
        proportions[mode]['Y0A0'] = base_A0[0]
        proportions[mode]['Y1A0'] = base_A0[1]
        proportions[mode]['Y0A1'] = base_A1[0]
        proportions[mode]['Y1A1'] = base_A1[1]

    #methods = ['nodebias', 'laftr','cfair_EO','cfair']
    #methods = ['nodebias','laftr', 'cfair', 'cfair_EO','cfair_ours']
    
    methods = ['nodebias','fair','cfair', 'laftr', 'cfair_EO']
    #methods = ['nodebias', 'laftr', 'cfair', 'cfair_EO', 'cfair_ours']
    lambdas = [1.0]

    trials = 1
    results = {}
    summary = {}
    summary['transfer'] = {} 
    summary['target_unfair'] = {} 
    for n in range(trials):
        for method in methods:
            for lamb in lambdas:
                m_lamb = f'{method}_lamb={lamb}'
                if n==0:
                    results[m_lamb] = {}
                
                args.epochs = 50
                
                t1 = time.time()
                print(f'====== Trial={n}|lamb={lamb}|method={method}========')
                performance_cls, performance_tf = train_nets(args, loaders, method, proportions, lamb=lamb)
                print(f'>>>TOOK {round(time.time()-t1,3)} secs')
                for key in performance_cls.keys():
                    if key not in results[m_lamb].keys():
                        results[m_lamb][key] = [performance_cls[key]]
                    else:
                        results[m_lamb][key].append(performance_cls[key])
        
                summary['transfer'][method] = performance_tf

    # target-unfair
    performance_tg_unfair = naive_train_nets(args, loaders)
    summary['target_unfair']['nodebias'] = performance_tg_unfair

    for method in methods:
        for lamb in lambdas:
            m_lamb = f'{method}_lamb={lamb}'
            summary[m_lamb]={}
            for key in results[m_lamb].keys():
                tmp_perform = results[m_lamb][key]
                tmp_perform = np.array(tmp_perform)

                summary[m_lamb][key] = [round(tmp_perform.mean(), 5), round(tmp_perform.std(), 5)]

    save_txt = f'transfer_results_{args.dataset}.txt'
    with open(save_txt, 'w') as f: 
        for method in summary.keys():
            f.write(f'========== Method >> {method} =========\n')
            for key in summary[method].keys():
                if key != 'Lamb':
                    f.write(f'{key:10}:{summary[method][key]}\n ')


    # save json
    import json
    save_json = f'results_{args.dataset}_all_lambda_1_new.json'
    with open(save_json,'w') as f:
        json.dump(summary, f)





def train_nets(args, loaders, method, prop, lamb):

    # define networks
    net_classifier = model_cfair.Classifier(args.dim_input, args.dim_hid_cls, include_A=args.include_A).cuda()
    net_adversary0 = model_cfair.Adversary(args.dim_hid_cls, args.dim_hid_adv).cuda()
    net_adversary1 = model_cfair.Adversary(args.dim_hid_cls, args.dim_hid_adv).cuda()

    # define optimizers
    optim_cls = optim.Adadelta(net_classifier.parameters(), lr=1.0) 
    optim_adv = optim.Adadelta(list(net_adversary0.parameters()) + list(net_adversary1.parameters()), lr=1.0) 
    optimizer = optim.Adadelta(list(net_classifier.parameters())+list(net_adversary0.parameters()) + list(net_adversary1.parameters()), lr=1.0) 
    
    # criterion
    ce = nn.CrossEntropyLoss()
    l1 = nn.L1Loss()
    sig = nn.Sigmoid()

    # global iteration 
    global_iter = 0

    # writer = SummaryWriter('tf_logs_cfair')
    # configure(f'tf_logs_cfair/{method}', flush_secs = 10)

    weight_y = 1.0 / torch.tensor([prop['train']['Y0'], prop['train']['Y1']]).cuda()
    weight_y_a0 = 1.0 / torch.tensor([prop['train']['Y0A0'],prop['train']['Y1A0']]).cuda()
    weight_y_a1 = 1.0 / torch.tensor([prop['train']['Y0A1'],prop['train']['Y1A1']]).cuda()

    for epoch in range(args.epochs): 
        # if epoch>2:
        #     break
        # start training
        net_classifier.train()
        net_adversary0.train()
        net_adversary1.train()

        for x, y, a, _ in loaders['train_loader']:
            x = x.cuda()
            y = y.cuda()
            a = a.cuda()

            # max h', h''
            # encode into hidden vectors
            # h_vec, _ = net_classifier(x)
            if method is 'nodebias':

                h_vec, logits_y = net_classifier(x)
                loss_cls = F.nll_loss(logits_y, y).mean()

                optim_cls.zero_grad()
                loss_cls.backward()
                optim_cls.step()

            else:
                # min h, g
                # feed forward
                h_vec, logits_y = net_classifier(x)

                rev_h_vec = model_cfair.grad_reverse(h_vec)
                
                if method is 'cfair':                
                    # adversary
                    logits_a_y0 = net_adversary0(rev_h_vec[y==0])
                    logits_a_y1 = net_adversary1(rev_h_vec[y==1])

                    loss_y = F.nll_loss(logits_y, y, weight=weight_y.float())
                    loss_adv_cls = 0.5 * F.nll_loss(logits_a_y0, a[y==0], weight=weight_y_a0.float()) \
                            + 0.5 * F.nll_loss(logits_a_y1, a[y==1], weight = weight_y_a1.float())
                
                elif method is 'cfair_EO':
                    # adversary
                    logits_a_y0 = net_adversary0(rev_h_vec[y==0])
                    logits_a_y1 = net_adversary1(rev_h_vec[y==1])

                    loss_y = F.nll_loss(logits_y, y)
                    loss_adv_cls = 0.5 * F.nll_loss(logits_a_y0, a[y==0], weight=weight_y_a0.float()) \
                            + 0.5 * F.nll_loss(logits_a_y1, a[y==1], weight = weight_y_a1.float())

                elif method is 'cfair_ours':
                    # adversary
                    logits_a_y0 = net_adversary0(rev_h_vec[y==0])
                    logits_a_y1 = net_adversary1(rev_h_vec[y==1])

                    loss_y = F.nll_loss(logits_y, y)
                    loss_adv_cls = 0.5 * F.nll_loss(logits_a_y0, a[y==0]) \
                            + 0.5 * F.nll_loss(logits_a_y1, a[y==1])

                elif method is 'fair':
                    # adversary
                    loss_y = F.nll_loss(logits_y, y)
                    logits_a_y0 = net_adversary0(rev_h_vec)

                    loss_adv_cls = F.nll_loss(logits_a_y0, a)

                elif method is 'laftr':
                    loss_y = F.nll_loss(logits_y, y)
                    logits_a_y0 = net_adversary0(rev_h_vec)[:,0]
                    loss_adv_cls = 0.0
                    for label in range(2):
                        for attribute in range(2):
                            idx = (y==label) & (a==attribute)
                            loss_adv_cls += l1(torch.exp(logits_a_y0[idx]), a[idx])
                    loss_adv_cls /= 4
                else:
                    print(method)
                
                if loss_y.mean() != loss_y.mean():
                    a = 1

                loss_y = loss_y.mean()
                loss_adv_cls = loss_adv_cls.mean()
                loss_cls = loss_y + lamb * loss_adv_cls
                
                optimizer.zero_grad()
                loss_cls.backward()
                optimizer.step()

        Y_hats = np.empty(0)
        A_hats = np.empty(0)
        Ys = np.empty(0)
        As = np.empty(0)

        net_classifier.eval()
        net_adversary0.eval()
        net_adversary1.eval()


        for x, y, a, _ in loaders['test_loader']:
            with torch.no_grad():
                x = x.cuda()
                y = y.cuda()
                a = a.cuda()
                #a_test = a.long().cuda().squeeze()

                # max h', h''
                # encode into hidden vectors
                h_vec, logits_y = net_classifier(x)
                rev_h_vec = model_cfair.grad_reverse(h_vec) 

                # adversary
                logits_a_y0 = net_adversary0(rev_h_vec)
                logits_a_y1 = net_adversary1(rev_h_vec)

                out_y = torch.argmax(logits_y, dim=-1)
                
                Y_hats = np.concatenate((Y_hats, out_y.cpu().numpy()))
                Ys = np.concatenate((Ys, y.cpu().numpy()))
                As = np.concatenate((As, a.cpu().numpy()))

        # err1, err2 w.r.t A. 
        Joint_Err = 1 - np.mean(Ys == Y_hats) 
        Err1 = np.mean(Y_hats[As==0] != Ys[As==0])
        Err2 = np.mean(Y_hats[As==1] != Ys[As==1])
        ErrGap = abs(Err1-Err2)
        JointErr = Err1+Err2
        
        cond_00 = np.mean(Y_hats[np.logical_and(As==0, Ys==0)])
        cond_10 = np.mean(Y_hats[np.logical_and(As!=0, Ys==0)])
        cond_01 = np.mean(Y_hats[np.logical_and(As==0, Ys!=0)])
        cond_11 = np.mean(Y_hats[np.logical_and(As!=0, Ys!=0)])

        DemoP = np.abs(np.mean(Y_hats[As==0]) - np.mean(Y_hats[As!=0]))
        EO_0 = np.abs(cond_00 - cond_10)
        EO_1 = np.abs(cond_01 - cond_11)
        EO = max(EO_0,EO_1)

        print(f'METHOD={method}, LAMBDA={lamb} || EPOCH={epoch} ErrGAP : {ErrGap:.3f} | JointErr : {JointErr:.3f} | EO : {EO:.3f} | DP : {DemoP:.3f}')

    performance_cls = {'ErrGap':ErrGap, 'JointErr':JointErr, 'EO':EO, 'DemoP':DemoP,'Lamb':lamb}

    # now transfer learning
    performance_transfer = {}

    # WE HAVE 10 different pcgs. 
    for pcg_idx in range(10):
        net_transfer_cls = model_cfair.TransferNet(args.dim_hid_cls).cuda()  
        optim_transfer = optim.Adadelta(net_transfer_cls.parameters(), lr=1.0) 
        for epoch in range(30): 
            # start training
            net_classifier.train()
            net_transfer_cls.train()

            for x, _, a, pcg in loaders['tf_train_loader']:
                x = x.cuda()
                a = a.cuda()
                pcg = pcg[:,pcg_idx].cuda()

                h_vec, _  = net_classifier(x)
                logits_pcg = net_transfer_cls(h_vec.detach())
                loss_cls = F.nll_loss(logits_pcg, pcg)

                optim_transfer.zero_grad()
                loss_cls.backward()
                optim_transfer.step()

            Y_hats = np.empty(0)
            A_hats = np.empty(0)
            Ys = np.empty(0)
            As = np.empty(0)
        
            # transfer learning validation
            net_classifier.eval()
            net_transfer_cls.eval()
            for x, _, a, pcg in loaders['tf_test_loader']:
                x = x.cuda()
                a = a.cuda()
                pcg = pcg[:,pcg_idx].cuda()

                h_vec, _  = net_classifier(x)
                logits_pcg = net_transfer_cls(h_vec)
                out_pcg = torch.argmax(logits_pcg, dim=-1)

                Y_hats = np.concatenate((Y_hats, out_pcg.cpu().numpy()))
                Ys = np.concatenate((Ys, pcg.cpu().numpy()))
                As = np.concatenate((As, a.cpu().numpy()))

            Joint_Err = 1 - np.mean(Ys == Y_hats) 
            Err1 = np.mean(Y_hats[As==0] != Ys[As==0])
            Err2 = np.mean(Y_hats[As==1] != Ys[As==1])
            ErrGap = abs(Err1-Err2)
            JointErr = Err1+Err2
            
            cond_00 = np.mean(Y_hats[np.logical_and(As==0, Ys==0)])
            cond_10 = np.mean(Y_hats[np.logical_and(As!=0, Ys==0)])
            cond_01 = np.mean(Y_hats[np.logical_and(As==0, Ys!=0)])
            cond_11 = np.mean(Y_hats[np.logical_and(As!=0, Ys!=0)])

            DemoP = np.abs(np.mean(Y_hats[As==0]) - np.mean(Y_hats[As!=0]))
            EO_0 = np.abs(cond_00 - cond_10)
            EO_1 = np.abs(cond_01 - cond_11)
            EO = max(EO_0,EO_1)

            print(f'TRANSFER[{pcg_idx}] : METHOD={method}, LAMBDA={lamb} || EPOCH={epoch} ErrGAP : {ErrGap:.3f} | JointErr : {JointErr:.3f} | EO : {EO:.3f} | DP : {DemoP:.3f}')
        performance_transfer[pcg_idx] = {'EO':EO, 'JointErr':JointErr}

     
    return performance_cls, performance_transfer

def naive_train_nets(args, loaders):
    # now transfer learning
    performance = {}

    # WE HAVE 10 different pcgs. 
    for pcg_idx in range(10):
        net_naive = model_cfair.Classifier(args.dim_input, args.dim_hid_cls).cuda()  
        optimizer = optim.Adadelta(net_naive.parameters(), lr=1.0) 
        for epoch in range(50): 
            # start training
            net_naive.train()

            for x, _, a, pcg in loaders['tf_train_loader']:
                x = x.cuda()
                a = a.cuda()
                pcg = pcg[:,pcg_idx].cuda()

                _, logits_pcg  = net_naive(x)
                loss_cls = F.nll_loss(logits_pcg, pcg)

                optimizer.zero_grad()
                loss_cls.backward()
                optimizer.step()

            Y_hats = np.empty(0)
            A_hats = np.empty(0)
            Ys = np.empty(0)
            As = np.empty(0)
        
            # transfer learning validation
            net_naive.eval()
            for x, _, a, pcg in loaders['tf_test_loader']:
                x = x.cuda()
                a = a.cuda()
                pcg = pcg[:,pcg_idx].cuda()

                _, logits_pcg = net_naive(x)
                out_pcg = torch.argmax(logits_pcg, dim=-1)

                Y_hats = np.concatenate((Y_hats, out_pcg.cpu().numpy()))
                Ys = np.concatenate((Ys, pcg.cpu().numpy()))
                As = np.concatenate((As, a.cpu().numpy()))

            Joint_Err = 1 - np.mean(Ys == Y_hats) 
            Err1 = np.mean(Y_hats[As==0] != Ys[As==0])
            Err2 = np.mean(Y_hats[As==1] != Ys[As==1])
            ErrGap = abs(Err1-Err2)
            JointErr = Err1+Err2
            
            cond_00 = np.mean(Y_hats[np.logical_and(As==0, Ys==0)])
            cond_10 = np.mean(Y_hats[np.logical_and(As!=0, Ys==0)])
            cond_01 = np.mean(Y_hats[np.logical_and(As==0, Ys!=0)])
            cond_11 = np.mean(Y_hats[np.logical_and(As!=0, Ys!=0)])

            DemoP = np.abs(np.mean(Y_hats[As==0]) - np.mean(Y_hats[As!=0]))
            EO_0 = np.abs(cond_00 - cond_10)
            EO_1 = np.abs(cond_01 - cond_11)
            EO = max(EO_0,EO_1)

            print(f'METHOD=TARGET_UNFAIR[{pcg_idx}], PCG_IDX={pcg_idx} || EPOCH={epoch} ErrGAP : {ErrGap:.3f} | JointErr : {JointErr:.3f} | EO : {EO:.3f} | DP : {DemoP:.3f}')
        performance[pcg_idx] = {'EO':EO, 'JointErr':JointErr}

    return performance

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--dataset', type=str, default='adult')
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--include_A', type=bool, default=False)
    args = parser.parse_args()

    main(args)

