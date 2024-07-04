import sys

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
import argparse
import os
from torch.utils.data import Subset

from utils import fix_random_seed, get_feature_extractor_model
from data.closed_set import get_in_testing_loader
from data.open_set import get_out_testing_datasets
from pgd_attack import attack_pgd
from models.DCGAN import Generator_fea, Discriminator_fea, wrapper_fea, Generator_pix, Discriminator_pix, weights_init
from models.preact_resnet import c10_preact_resnet

os.environ['TORCH_HOME'] = 'models/'


def diffscale(diff):
    diffmin = np.min(diff)
    diffmax = np.max(diff)
    diff = (diff - diffmin) / (diffmax - diffmin)
    return diff


def save_image():
    if args.imgdir != "None":
        max_index = np.argmax(distance_x_xc)
        max_x_test = x_test[max_index]
        max_x_test_ood = x_test_ood[max_index]

        if not os.path.exists(f'{args.imgdir}/max(x,xc)/'):
            os.makedirs(f'{args.imgdir}/max(x,xc)')
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(max_x_test.transpose(1, 2, 0).reshape(32, 32, 3))
        axes[0].set_axis_off()
        axes[0].set_title(f"x_test")
        axes[1].imshow(max_x_test_ood.transpose(1, 2, 0).reshape(32, 32, 3))
        axes[1].set_axis_off()
        axes[1].set_title(f"x_test_ood")
        plt.savefig(f"{args.imgdir}/max(x,xc)/figure_max_{args.comps}_comps.png")
        plt.close()

        min_index = np.argmin(distance_x_xc)
        min_x_test = x_test[min_index]
        min_x_test_ood = x_test_ood[min_index]

        if not os.path.exists(f'{args.imgdir}/min(x,xc)/'):
            os.makedirs(f'{args.imgdir}/min(x,xc)')
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(min_x_test.transpose(1, 2, 0).reshape(32, 32, 3))
        axes[0].set_axis_off()
        axes[0].set_title(f"x_test")
        axes[1].imshow(min_x_test_ood.transpose(1, 2, 0).reshape(32, 32, 3))
        axes[1].set_axis_off()
        axes[1].set_title(f"x_test_ood")
        plt.savefig(f"{args.imgdir}/min(x,xc)/figure_min_{args.comps}_comps.png")
        plt.close()

        max_index = np.argmax(distance_x_xca)
        max_x_test = x_test[max_index]
        max_x_test_ood_adv = x_test_ood_adv[max_index]
        if not os.path.exists(f'{args.imgdir}/max(x,xca)/'):
            os.makedirs(f'{args.imgdir}/max(x,xca)')
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(max_x_test.transpose(1, 2, 0).reshape(32, 32, 3))
        axes[0].set_axis_off()
        axes[0].set_title(f"x_test")
        axes[1].imshow(max_x_test_ood_adv.transpose(1, 2, 0).reshape(32, 32, 3))
        axes[1].set_axis_off()
        axes[1].set_title(f"x_test_ood_adv")
        plt.savefig(f"{args.imgdir}/max(x,xca)/figure_max_{args.comps}_comps.png")
        plt.close()

        # min distance image

        min_index = np.argmin(distance_x_xca)
        min_x_test = x_test[min_index]
        min_x_test_ood_adv = x_test_ood_adv[min_index]

        if not os.path.exists(f'{args.imgdir}/min(x,xca)/'):
            os.makedirs(f'{args.imgdir}/min(x,xca)')
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(min_x_test.transpose(1, 2, 0).reshape(32, 32, 3))
        axes[0].set_axis_off()
        axes[0].set_title(f"x_test")
        axes[1].imshow(min_x_test_ood_adv.transpose(1, 2, 0).reshape(32, 32, 3))
        axes[1].set_axis_off()
        axes[1].set_title(f"x_test_ood_adv")
        plt.savefig(f"{args.imgdir}/min(x,xca)/figure_min_{args.comps}_comps.png")
        plt.close()

        if not os.path.exists(f'{args.imgdir}/{args.comps}'):
            os.makedirs(f'{args.imgdir}/{args.comps}')

        for i in range(100):
            diff = x_test_adv[i] - x_test[i]
            diff = diffscale(diff)

            diffood = x_test_ood_adv[i] - x_test_ood[i]
            diffood = diffscale(diffood)
            fig, axes = plt.subplots(2, 3)
            axes[0, 0].imshow(x_test[i].transpose(1, 2, 0).reshape(32, 32, 3))
            axes[0, 0].set_axis_off()
            axes[0, 0].set_title(f"x_test")
            axes[0, 2].imshow(x_test_adv[i].transpose(1, 2, 0).reshape(32, 32, 3))
            axes[0, 2].set_axis_off()
            axes[0, 2].set_title(f"x_test_adv")
            axes[1, 0].imshow(x_test_ood[i].transpose(1, 2, 0).reshape(32, 32, 3))
            axes[1, 0].set_axis_off()
            axes[1, 0].set_title(f"x_test_ood")
            axes[1, 2].imshow(x_test_ood_adv[i].transpose(1, 2, 0).reshape(32, 32, 3))
            axes[1, 2].set_axis_off()
            axes[1, 2].set_title(f"x_test_ood_adv")
            axes[0, 1].imshow(diff.transpose(1, 2, 0).reshape(32, 32, 3))
            axes[0, 1].set_axis_off()
            axes[0, 1].set_title("diff(x,xa)")
            axes[1, 1].imshow(diffood.transpose(1, 2, 0).reshape(32, 32, 3))
            axes[1, 1].set_axis_off()
            axes[1, 1].set_title("diff(xc,xca)")

            plt.savefig(f"{args.imgdir}/{args.comps}/figure_{i}.png", bbox_inches='tight')
            plt.close()

# get args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='fea', type=str, choices={'fea', 'pix'})
    parser.add_argument('--targetlabel', default='all', type=str)
    parser.add_argument('--imgdir', default='test', type=str)
    parser.add_argument('--comps', default='100', type=int)
    parser.add_argument('--training_type', default='adv', type=str, choices={'clean', 'adv'})
    parser.add_argument('--in_dataset', default='cifar10', type=str, choices={'cifar10', 'cifar100', 'TI'})
    parser.add_argument("--out_datasets", nargs='+',
                        default=['mnist', 'tiny_imagenet', 'places', 'LSUN', 'iSUN', 'birds', 'flowers', 'coil'])

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--eps', default=8 / 255, type=float)
    parser.add_argument('--attack_iters', default=100, type=int)

    parser.add_argument('--run_name', default='test', type=str)
    parser.add_argument('--seed', default=0, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model_type = args.model_type
    training_type = args.training_type
    in_dataset = args.in_dataset
    out_names = args.out_datasets

    batch_size = args.batch_size
    eps = args.eps
    epsilons = np.array([0, eps])
    attack_iters = args.attack_iters

    run_name = args.run_name
    test_type = 'best_'
    save_dir = 'checkpoints/'
    print('Run name:', run_name)

    if in_dataset == 'TI' and 'tiny_imagenet' in out_names:
        out_names.remove('tiny_imagenet')

    # set random seed
    seed = args.seed
    fix_random_seed(seed)

    # define deture extractor model
    model = get_feature_extractor_model(training_type, in_dataset)

    # in dataset
    testloader = get_in_testing_loader(in_dataset, batch_size)

    # out datasets
    out_names, out_datasets = get_out_testing_datasets(out_names, comps=args.comps, targetlabel=args.targetlabel)

    print('Out datasets:', out_names)

    # Model DCGAN
    # Number of channels in the training images. For color images this is 3
    if model_type == 'fea':
        nc = 512
    elif model_type == 'pix':
        nc = 3

    # classifier = c10_preact_resnet(10)
    #
    # state_dict = torch.load(os.path.join(save_dir, 'classifier.pt'))
    #
    # new_state_dict = {}
    # for key, value in state_dict['model_state_dict'].items():
    #    new_key = key.replace("module.0.", "")  # Remove the "module.0" prefix
    #    new_state_dict[new_key] = value
    # print(new_state_dict.keys())
    #
    # classifier.load_state_dict(new_state_dict)


    ndf = 64

    # Number of GPUs available.
    ngpu = 1

    if model_type == 'fea':
        netD = Discriminator_fea(ngpu=ngpu, nc=nc, ndf=ndf).to(device)

        forward_pass = wrapper_fea(model, netD)

    elif model_type == 'pix':
        netD = Discriminator_pix(ngpu=ngpu, nc=nc, ndf=ndf).to(device)

        forward_pass = netD

    # load model
    print('\n', test_type)
    netD.load_state_dict(torch.load(os.path.join(save_dir, 'DNet_' + test_type + run_name)))
    netD.eval()

    scores_in = [[] for i in epsilons]
    scores_out = [[[] for j in epsilons] for i in out_datasets]

    x_test = []
    x_test_adv = []
    pred_in = []
    pred_a_in = []
    pred_out = []
    pred_a_out = []

    # scores in
    for i, eps in enumerate(epsilons):

        alpha = 2.5 * eps / attack_iters

        for (x, y) in tqdm.tqdm(testloader, desc=in_dataset + "_" + str(round(eps, 3))):
            x = x.to(device)  # tesztképek?

            if eps == 0:
                delta = torch.zeros_like(x)
            else:
                delta = attack_pgd(forward_pass, x, torch.ones_like(y, dtype=torch.float32).to(device), epsilon=eps,
                                   alpha=alpha, attack_iters=attack_iters)
                x_test.append(x.detach().cpu())
                x_test_adv.append((x + delta).detach().cpu())



            output = forward_pass(x + delta).view(-1)  # támadott tesztkép?
            scores_in[i] += output.cpu().detach().tolist()
            #pred_in.append(classifier.forward(x).detach().cpu())
            #pred_a_in.append(classifier.forward(x + delta).detach().cpu())


    x_test = np.concatenate(x_test, axis=0)
    x_test_adv = np.concatenate(x_test_adv, axis=0)



    #pred_in = np.concatenate(pred_in, axis=0)
    #pred_a_in = np.concatenate(pred_a_in, axis=0)

    x_test_ood = []
    x_test_ood_adv = []
    # scores out
    for i, dataset in enumerate(out_datasets):

        for j, eps in enumerate(epsilons):
            alpha = 2.5 * eps / attack_iters

            testloader_out = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            for (x, y) in tqdm.tqdm(testloader_out, desc=args.out_datasets[i] + "_" + str(round(eps, 3))):
                x = x.to(device)  # ood kép, compressed

                if eps == 0:
                    delta = torch.zeros_like(x)
                else:
                    delta = attack_pgd(forward_pass, x, torch.zeros_like(y, dtype=torch.float32).to(device),
                                       epsilon=eps,
                                       alpha=alpha, attack_iters=attack_iters)
                    x_test_ood.append(x.detach().cpu())
                    x_test_ood_adv.append((x + delta).detach().cpu())

                output = forward_pass(x + delta).view(-1)
                scores_out[i][j] += output.cpu().detach().tolist()
                #pred_out.append(classifier(x).detach().cpu())
                #pred_a_out.append(classifier(x + delta).detach().cpu())


    x_test_ood = np.concatenate(x_test_ood, axis=0)
    x_test_ood_adv = np.concatenate(x_test_ood_adv, axis=0)
    #pred_out = np.concatenate(pred_out, axis=0)
    #pred_a_out = np.concatenate(pred_a_out, axis=0)

    distance_x_xc = np.max(np.abs(x_test - x_test_ood), axis=(1, 2, 3))
    distance_x_xca = np.max(np.abs(x_test - x_test_ood_adv), axis=(1, 2, 3))

    # save images



    # auc
    for i, score_out_dataset in enumerate(scores_out):

        print('\ndataset:', out_names[i])

        print('\njust in attacked')
        score_out = score_out_dataset[0]
        for k, score_in in enumerate(scores_in):
            onehots = np.array([1] * len(score_out) + [0] * len(score_in))
            scores = np.concatenate([score_out, score_in], axis=0)
            auroc = roc_auc_score(onehots, -scores)
            print('eps=', epsilons[k], ':', auroc)

        print('\njust out attacked')
        score_in = scores_in[0]
        for k, score_out in enumerate(score_out_dataset):
            onehots = np.array([1] * len(score_out) + [0] * len(score_in))
            scores = np.concatenate([score_out, score_in], axis=0)
            auroc = roc_auc_score(onehots, -scores)
            print('eps=', epsilons[k], ':', auroc)

        print('\nboth attacked')
        for k in range(len(scores_in)):
            score_in = scores_in[k]
            score_out = score_out_dataset[k]

            onehots = np.array([1] * len(score_out) + [0] * len(score_in))
            scores = np.concatenate([score_out, score_in], axis=0)
            auroc = roc_auc_score(onehots, -scores)
            print('eps=', epsilons[k], ':', auroc)
