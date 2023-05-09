import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import os
import torch
import time

# from dataset import get_loader
from .dagmm import DAGMM
from tqdm import tqdm

has_threshold = True


def load_model(hyp):
    model = DAGMM(hyp)
    try:
        ckpt_path = os.path.join(
            hyp["save_dir"], "{}_dagmm.pth".format(hyp["checkpoint"])
        )
        model.load_state_dict(torch.load(ckpt_path))
        print("success to load model")
    except Exception as e:
        print("Failed to load model: %s" % (e))
        exit(1)

    return model


def compute_threshold(model, train_loader, len_data):
    energies = np.zeros(shape=(len_data))
    step = 0
    energy_interval = 50

    with torch.no_grad():
        for x, y in train_loader:

            enc, dec, z, gamma = model(x)
            m_prob, m_mean, m_cov = model.get_gmm_param(gamma, z)

            for i in range(z.shape[0]):
                zi = z[i].unsqueeze(1)
                sample_energy = model.sample_energy(
                    m_prob, m_mean, m_cov, zi, gamma.shape[1], gamma.shape[0]
                )

                energies[step] = sample_energy.detach().item()
                step += 1

            if step % energy_interval == 0:
                print("Iteration: %d    sample energy: %.4f" % (step, sample_energy))

    threshold = np.percentile(energies, 80)
    print("threshold: %.4f" % (threshold))

    return threshold


def decision_function(hyp, X_test):
    model = load_model(hyp)
    model.eval()

    len_data = len(X_test)
    scores = np.zeros(shape=(len_data, 1))

    with torch.no_grad():
        for idx, x in enumerate(X_test):
            
            enc, dec, z, gamma = model(x.unsqueeze(0))
            # print(x.shape, z.shape, gamma.shape)
            
            m_prob, m_mean, m_cov = model.get_gmm_param(gamma, z)
            zi = z[0].unsqueeze(1)
            sample_energy = model.sample_energy(
                m_prob, m_mean, m_cov, zi, gamma.shape[1], gamma.shape[0]
            )
            se = sample_energy.detach().item()
            scores[idx] = se
            break


    return scores


def main(hyp, test_loader, val_loader=None, has_threshold=True):
    model = load_model(hyp)
    model.eval()

    # train_loader, len_data = get_loader(hyp, mode="train")
    # if has_threshold == False:
    #     threshold = compute_threshold(model, train_loader, len_data)
    # else:
    #     # threshold = -6.2835
    #     threshold = -0.6870

    # print("threshold: ", threshold)

    m_probs = []
    m_means = []
    m_covs = []
    if val_loader is not None:
        for x, y in val_loader:
            enc, dec, z, gamma = model(x)
            m_prob, m_mean, m_cov = model.get_gmm_param(gamma, z)
            m_probs.append(m_prob)
            m_means.append(m_mean)
            m_covs.append(m_cov)
        
        m_probs = torch.cat(m_probs, dim=0).mean(dim=0, keepdim=True)
        m_means = torch.cat(m_means, dim=0).mean(dim=0, keepdim=True)
        m_covs = torch.cat(m_covs, dim=0).mean(dim=0, keepdim=True)
        print(m_probs, m_means, m_covs)

    # test_loader, len_data = get_loader(hyp, mode="test")
    len_data = len(test_loader.dataset)
    scores = np.zeros(shape=(len_data, 2))
    step = 0
    idx = 0
    with torch.no_grad():
        for x, y in test_loader:

            enc, dec, z, gamma = model(x)

            if val_loader is None:
                m_prob, m_mean, m_cov = model.get_gmm_param(gamma, z)
            # print(x.shape, z.shape, gamma.shape)
            # sample_energy = model.sample_energy(
            #         m_prob, m_mean, m_cov, z, gamma.shape[1], gamma.shape[0]
            #     )
            # se = sample_energy.cpu().item()
            # scores[idx:idx+len(sample_energy), 0] = y.item()
            # scores[idx:idx+len(sample_energy), 1] = se
            # idx += len(sample_energy)
            for i in range(z.shape[0]):
                zi = z[i].unsqueeze(1)
                sample_energy = model.sample_energy(
                    m_prob, m_mean, m_cov, zi, gamma.shape[1], gamma.shape[0]
                )
                se = sample_energy.detach().item()

                # scores[step] = [int(y[i]), int(se > threshold)]
                scores[step] = [int(y[i]), se]

                step += 1

    # accuracy = accuracy_score(scores[:, 0], scores[:, 1])
    # precision, recall, fscore, support = precision_recall_fscore_support(
    #     scores[:, 0], scores[:, 1], average="binary"
    # )
    # print(
    #     "Accuracy: %.4f  Precision: %.4f  Recall: %.4f  F-score: %.4f"
    #     % (accuracy, precision, recall, fscore)
    # )

    return scores


if __name__ == "__main__":

    hyp = {
        "input_dim": 118,
        "hidden1_dim": 60,
        "hidden2_dim": 30,
        "hidden3_dim": 10,
        "zc_dim": 1,
        "emb_dim": 10,
        "n_gmm": 2,
        "dropout": 0.5,
        "lambda1": 0.1,
        "lambda2": 0.005,
        "lr": 1e-4,
        "batch_size": 128,
        "epochs": 20,
        "print_iter": 300,
        "savestep_epoch": 2,
        "save_path": "./models/",
        "data_dir": "../dagmm-master/kdd_cup.npz",
        "img_dir": "./result/",
        "ratio": 0.8,
    }

    main(hyp)
