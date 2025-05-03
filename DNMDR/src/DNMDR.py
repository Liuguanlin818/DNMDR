import dill
import numpy as np
import argparse
from collections import defaultdict
from sklearn.metrics import jaccard_score
from torch.optim import Adam
import os
import torch
import time
from models import DNMDR
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, buildMPNN
import torch.nn.functional as F
import sys
import pickle

torch.manual_seed(1203)
np.random.seed(2048)
model_name = "DNMDR"
resume_path = "Epoch_32_TARGET_0.06_JA_0.5241_DDI_0.06058.model"

if not os.path.exists(os.path.join("saved", model_name)):
    os.makedirs(os.path.join("saved", model_name))

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--Test", action="store_true", default=False, help="test mode")
parser.add_argument("--model_name", type=str, default=model_name, help="model name")
parser.add_argument("--resume_path", type=str, default=resume_path, help="resume path")
parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument("--target_ddi", type=float, default=0.06, help="target ddi")
parser.add_argument("--kp", type=float, default=0.05, help="coefficient of P signal")
parser.add_argument("--dim", type=int, default=64, help="dimension")
parser.add_argument("--cuda", type=int, default=1, help="which cuda")
args = parser.parse_args()


# evaluate
def eval(model, data_eval, voc_size, epoch):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, adm in enumerate(input):
            target_output, _ = model(input[: adm_idx + 1])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prod
            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            # prediction med set
            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint("\rtest step: {} / {}".format(step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path="../data/output/ddi_A_final.pkl")

    llprint(
        "\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
            ddi_rate,
            np.mean(ja),  # 求取均值
            np.mean(prauc),
            np.mean(avg_p),
            np.mean(avg_r),
            np.mean(avg_f1),
            med_cnt / visit_cnt,
        )
    )

    return (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        med_cnt / visit_cnt,
    )


def count_conditional_prob_dp(seqex_list, output_path, train_key_set=None):
    dx_freqs = {}
    proc_freqs = {}
    med_freqs = {}
    dm_freqs = {}
    pm_freqs = {}
    total_visit = 0
    for seqex in seqex_list:
        if total_visit % 1000 == 0:
            sys.stdout.write('Visit count: %d\r' % total_visit)
            sys.stdout.flush()

        if train_key_set is not None and seqex not in train_key_set:
            total_visit += len(seqex)
            continue

        for key in seqex:
            dx_ids = key[0]
            proc_ids = key[1]
            med_ids = key[2]

            for dx in dx_ids:

                if dx not in dx_freqs:
                    dx_freqs[dx] = 0

                dx_freqs[dx] += 1


            for proc in proc_ids:
                if proc not in proc_freqs:
                    proc_freqs[proc] = 0
                proc_freqs[proc] += 1


            for med in med_ids:
                if med not in med_freqs:
                    med_freqs[med] = 0
                med_freqs[med] += 1


            for dx in dx_ids:
                for med in med_ids:
                    dm = str(dx) + ',' + str(med)
                    if dm not in dm_freqs:
                        dm_freqs[dm] = 0
                    dm_freqs[dm] += 1


            for proc in proc_ids:
                for med in med_ids:
                    pm = str(proc) + ',' + str(med)
                    if pm not in pm_freqs:
                        pm_freqs[pm] = 0
                    pm_freqs[pm] += 1

            total_visit += 1


    dx_probs = dict([(k, v / float(total_visit)) for k, v in dx_freqs.items()])
    proc_probs = dict([(k, v / float(total_visit)) for k, v in proc_freqs.items()])
    med_probs = dict([(k, v / float(total_visit)) for k, v in med_freqs.items()])
    dm_probs = dict([(k, v / float(total_visit)) for k, v in dm_freqs.items()])
    pm_probs = dict([(k, v / float(total_visit)) for k, v in pm_freqs.items()])

    dm_cond_probs = {}
    md_cond_probs = {}
    for dx, dx_prob in dx_probs.items():
        for med, med_prob in med_probs.items():
            dm = str(dx) + ',' + str(med)
            md = str(med) + ',' + str(dx)
            if dm in dm_probs:
                dm_cond_probs[dm] = dm_probs[dm] / dx_prob
                md_cond_probs[md] = dm_probs[dm] / med_prob
            else:
                dm_cond_probs[dm] = 0.0
                md_cond_probs[md] = 0.0

    pm_cond_probs = {}
    mp_cond_probs = {}
    for proc, proc_prob in proc_probs.items():
        for med, med_prob in med_probs.items():
            pm = str(proc) + ',' + str(med)
            mp = str(med) + ',' + str(proc)

            if pm in pm_probs:
                pm_cond_probs[pm] = pm_probs[pm] / proc_prob
                mp_cond_probs[mp] = pm_probs[pm] / med_prob

            else:
                pm_cond_probs[pm] = 0.0
                mp_cond_probs[mp] = 0.0

    pickle.dump(dx_probs, open(output_path + '/dx_probs.empirical.p', 'wb'), -1)
    pickle.dump(proc_probs, open(output_path + '/proc_probs.empirical.p', 'wb'), -1)
    pickle.dump(med_probs, open(output_path + '/med_probs.empirical.p', 'wb'), -1)
    pickle.dump(dm_probs, open(output_path + '/dm_probs.empirical.p', 'wb'), -1)
    pickle.dump(dm_cond_probs, open(output_path + '/dm_cond_probs.empirical.p', 'wb'), -1)
    pickle.dump(md_cond_probs, open(output_path + '/md_cond_probs.empirical.p', 'wb'), -1)
    pickle.dump(pm_probs, open(output_path + '/pm_probs.empirical.p', 'wb'), -1)
    pickle.dump(pm_cond_probs, open(output_path + '/pm_cond_probs.empirical.p', 'wb'), -1)
    pickle.dump(mp_cond_probs, open(output_path + '/mp_cond_probs.empirical.p', 'wb'), -1)


def add_sparse_prior_guide_dp(seqex_list, stats_path, key_set=None):
    print('Loading conditional probabilities.')
    dm_cond_probs = pickle.load(open(stats_path + '/dm_cond_probs.empirical.p', 'rb'))
    md_cond_probs = pickle.load(open(stats_path + '/md_cond_probs.empirical.p', 'rb'))
    pm_cond_probs = pickle.load(open(stats_path + '/pm_cond_probs.empirical.p', 'rb'))
    mp_cond_probs = pickle.load(open(stats_path + '/mp_cond_probs.empirical.p', 'rb'))

    print('Adding prior guide.')
    total_visit = 0
    new_seqex_list = []

    for seqex in seqex_list:
        if total_visit % 1000 == 0:
            sys.stdout.write('Visit count: %d\r' % total_visit)
            sys.stdout.flush()

        if key_set is not None and seqex not in key_set:
            total_visit += len(seqex)
            continue

        for key in seqex:
            dx_ids = key[0]
            proc_idx = key[1]
            med_ids = key[2]

            indices_dpm = []
            values_dpm = []

            for i, dx in enumerate(dx_ids):
                for j, med in enumerate(med_ids):
                    dm = str(dx) + ',' + str(med)
                    indices_dpm.append((i, len(dx_ids) + len(proc_idx) + med))
                    prob = 0.0 if dm not in dm_cond_probs else dm_cond_probs[dm]
                    values_dpm.append(prob)

            for i, proc in enumerate(proc_idx):
                for j, med in enumerate(med_ids):
                    pm = str(proc) + ',' + str(med)
                    indices_dpm.append((len(dx_ids) + i, len(dx_ids) + len(proc_idx) + med))
                    prob = 0.0 if pm not in pm_cond_probs else pm_cond_probs[pm]
                    values_dpm.append(prob)

            for i, med in enumerate(med_ids):
                for j, dx in enumerate(dx_ids):
                    md = str(med) + ',' + str(dx)
                    indices_dpm.append((len(dx_ids) + len(proc_idx) + med, j))
                    prob = 0.0 if md not in md_cond_probs else md_cond_probs[md]
                    values_dpm.append(prob)

                for j, proc in enumerate(proc_idx):
                    mp = str(med) + ',' + str(proc)
                    indices_dpm.append((len(dx_ids) + len(proc_idx) + med, len(dx_ids) + j))
                    prob = 0.0 if mp not in mp_cond_probs else mp_cond_probs[mp]
                    values_dpm.append(prob)

            key.append(indices_dpm)
            key.append(values_dpm)

            total_visit += 1


def write_log(log_path, epoch, results):
    with open(log_path, 'a') as f:
        f.write(f"epoch {epoch + 1} --------------------------\n")
        f.write(f"DDI Rate: {results['ddi_rate']:.4f}, Jaccard: {results['ja']:.4f}, "
                f"PRAUC: {results['prauc']:.4f}, AVG_PRC: {results['avg_p']:.4f}, "
                f"AVG_RECALL: {results['avg_r']:.4f}, AVG_F1: {results['avg_f1']:.4f}, "
                f"AVG_MED: {results['avg_med']:.4f}"
                f"LOSS: {results['loss']:.4f}\n")
        f.write(f"best_Epoch: {results['best_epoch']}\n\n")


def main():
    # load data
    data_path = "../data/output/records_final.pkl"
    voc_path = "../data/output/voc_final.pkl"
    # 文件路径
    ehr_adj_path = '../data/output/ehr_adj_final.pkl'
    ddi_adj_path = "../data/output/ddi_A_final.pkl"
    ddi_mask_path = "../data/output/ddi_mask_H.pkl"
    molecule_path = "../data/output/atc3toSMILES.pkl"
    device = torch.device("cuda:0")
    # 加载文件
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, "rb"))
    ddi_mask_H = dill.load(open(ddi_mask_path, "rb"))
    data = dill.load(open(data_path, "rb"))
    molecule = dill.load(open(molecule_path, "rb"))

    voc = dill.load(open(voc_path, "rb"))
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point: split_point + eval_len]
    data_eval = data[split_point + eval_len:]
    # MPNN
    MPNNSet, N_fingerprint, average_projection = buildMPNN(molecule, med_voc.idx2word, 2, device)

    log_path = os.path.join("saved", model_name, "visit4_log.txt")
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))

    stats_path = '../data/train_stats'
    if not os.path.exists(stats_path):
        os.makedirs(stats_path)
    count_conditional_prob_dp(data, stats_path, data_train)

    add_sparse_prior_guide_dp(data, stats_path, data_train)
    add_sparse_prior_guide_dp(data, stats_path, data_eval)
    add_sparse_prior_guide_dp(data, stats_path, data_test)

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    model = DNMDR(
        voc_size,
        ehr_adj,
        ddi_adj,
        ddi_mask_H,
        MPNNSet,  #
        N_fingerprint,
        average_projection,
        emb_dim=args.dim,
        device=device,
    )
    model.to(device=device)
    optimizer = Adam(list(model.parameters()), lr=args.lr)

    # start iterations
    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    EPOCH = 50
    for epoch in range(EPOCH):
        tic = time.time()
        print("\nepoch {} --------------------------".format(epoch + 1))

        model.train()
        for step, input in enumerate(data_train):

            loss = 0
            for idx, adm in enumerate(input):

                seq_input = input[: idx + 1]
                loss_bce_target = np.zeros((1, voc_size[2]))
                loss_bce_target[:, adm[2]] = 1

                loss_multi_target = np.full((1, voc_size[2]), -1)
                for idx, item in enumerate(adm[2]):
                    loss_multi_target[0][idx] = item

                result, loss_ddi = model(seq_input)

                loss_bce = F.binary_cross_entropy_with_logits(result, torch.FloatTensor(loss_bce_target).to(device))
                loss_multi = F.multilabel_margin_loss(F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device))

                result = F.sigmoid(result).detach().cpu().numpy()[0]
                result[result >= 0.5] = 1
                result[result < 0.5] = 0
                y_label = np.where(result == 1)[0]
                current_ddi_rate = ddi_rate_score([[y_label]], path="../data/output/ddi_A_final.pkl")

                if current_ddi_rate <= args.target_ddi:
                    loss = 0.95 * loss_bce + 0.05 * loss_multi
                else:
                    beta = min(0, 1 + (args.target_ddi - current_ddi_rate) / args.kp)
                    loss = (
                            beta * (0.95 * loss_bce + 0.05 * loss_multi)
                            + (1 - beta) * loss_ddi
                    )

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            llprint("\rtraining step: {} / {}".format(step, len(data_train)))

        print()
        tic2 = time.time()
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, data_eval, voc_size, epoch)
        print("training time: {}, test time: {}".format(time.time() - tic, time.time() - tic2))

        history["ja"].append(ja)
        history["ddi_rate"].append(ddi_rate)
        history["avg_p"].append(avg_p)
        history["avg_r"].append(avg_r)
        history["avg_f1"].append(avg_f1)
        history["prauc"].append(prauc)
        history["med"].append(avg_med)

        if epoch >= 5:
            print("ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}".format(np.mean(history["ddi_rate"][-5:]),
                                                                       np.mean(history["med"][-5:]),
                                                                       np.mean(history["ja"][-5:]),
                                                                       np.mean(history["avg_f1"][-5:]),
                                                                       np.mean(history["prauc"][-5:]), ))


        results = {
            'ddi_rate': ddi_rate,
            'ja': ja,
            'prauc': prauc,
            'avg_p': avg_p,
            'avg_r': avg_r,
            'avg_f1': avg_f1,
            'avg_med': avg_med,
            'best_epoch': best_epoch,
            'loss': loss.item()
        }
        write_log(log_path, epoch, results)

        torch.save(
            model.state_dict(),
            open(
                os.path.join(
                    "saved",
                    args.model_name,
                    "Epoch_{}_TARGET_{:.2}_JA_{:.4}_DDI_{:.4}.model".format(
                        epoch, args.target_ddi, ja, ddi_rate
                    ),
                ),
                "wb",
            ),
        )

        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja = ja

        print("best_epoch: {}".format(best_epoch))

    dill.dump(
        history,
        open(
            os.path.join(
                "saved", args.model_name, "history_{}.pkl".format(args.model_name)
            ),
            "wb",
        ),
    )


if __name__ == "__main__":
    main()
