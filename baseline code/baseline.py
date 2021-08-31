import sys

from pathlib import Path
from datetime import timedelta

import dateutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

import os

from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import trange
from TaPR_pkg import etapr

#Setting environment
os.environ['KMP_DUPLICATE_LIB_OK']='True'

'''
###ORIGINAL###
#Input
WINDOW_GIVEN = 89
#Output
WINDOW_SIZE = 90

N_HIDDENS = 100
N_LAYERS = 3
BATCH_SIZE = 512

#Time window for sliding
stride = 10
'''

WINDOW_GIVEN = 79
WINDOW_SIZE = 80

#Optimal N_HIDDENS = 80
N_HIDDENS = 100
#Optimal N_LAYERS = 4
N_LAYERS = 3
BATCH_SIZE = 1536

#Time window for sliding => optimal stride : 12
stride = 8
epoch = 70

#Dataset Setting
TRAIN_DATASET = sorted([x for x in Path("235757_HAICon2021_dataset/train/").glob("*.csv")])
TEST_DATASET = sorted([x for x in Path("235757_HAICon2021_dataset/test/").glob("*.csv")])
VALIDATION_DATASET = sorted([x for x in Path("235757_HAICon2021_dataset/validation/").glob("*.csv")])

#Setting columns to timestame, id, field
TIMESTAMP_FIELD = "timestamp"
IDSTAMP_FIELD = 'id'
ATTACK_FIELD = "attack"

#StackedGRU model
class StackedGRU(torch.nn.Module):
    def __init__(self, n_tags):
        super().__init__()
        self.rnn = torch.nn.GRU(
            input_size=n_tags,
            hidden_size=N_HIDDENS,
            num_layers=N_LAYERS,
            bidirectional=True,
            dropout=0,
        )
        self.fc = torch.nn.Linear(N_HIDDENS * 2, n_tags)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(0, 1)  # (batch, seq, params) -> (seq, batch, params)
        self.rnn.flatten_parameters()
        outs, _ = self.rnn(x)
        out = self.fc(outs[-1])
        return x[0] + out

#Dataset Settings
class HaiDataset(Dataset):
    def __init__(self, timestamps, df, stride=1, attacks=None):
        self.ts = np.array(timestamps)
        self.tag_values = np.array(df, dtype=np.float32)
        self.valid_idxs = []
        for L in tqdm(range(len(self.ts) - WINDOW_SIZE + 1)):
            R = L + WINDOW_SIZE - 1
            if dateutil.parser.parse(self.ts[R]) - dateutil.parser.parse(self.ts[L]) == timedelta(seconds=WINDOW_SIZE - 1):
                self.valid_idxs.append(L)
        self.valid_idxs = np.array(self.valid_idxs, dtype=np.int32)[::stride]
        self.n_idxs = len(self.valid_idxs)
        #print(f"# of valid windows: {self.n_idxs}")
        if attacks is not None:
            self.attacks = np.array(attacks, dtype=np.float32)
            self.with_attack = True
        else:
            self.with_attack = False

    def __len__(self):
        return self.n_idxs

    def __getitem__(self, idx):
        i = self.valid_idxs[idx]
        last = i + WINDOW_SIZE - 1
        item = {"attack": self.attacks[last]} if self.with_attack else {}
        item["ts"] = self.ts[i + WINDOW_SIZE - 1]
        item["given"] = torch.from_numpy(self.tag_values[i : i + WINDOW_GIVEN])
        item["answer"] = torch.from_numpy(self.tag_values[last])
        return item

class Baseline:

#Training!!!!!!!!!!!!!!
    def Training(self, TRAIN_DATASET, stride, epoch):
        #Trainset to dataframe
        TRAIN_DF_RAW = Baseline.dataframe_from_csvs(TRAIN_DATASET)
        self.VALID_COLUMNS_IN_TRAIN_DATASET = TRAIN_DF_RAW.columns.drop([TIMESTAMP_FIELD])

        #Tagging min & max
        self.TAG_MIN = TRAIN_DF_RAW[self.VALID_COLUMNS_IN_TRAIN_DATASET].min()
        self.TAG_MAX = TRAIN_DF_RAW[self.VALID_COLUMNS_IN_TRAIN_DATASET].max()

        #Normalizing
        TRAIN_DF = Baseline.normalize(TRAIN_DF_RAW[self.VALID_COLUMNS_IN_TRAIN_DATASET], self.TAG_MIN, self.TAG_MAX).ewm(alpha=0.9).mean()

        #boundary_checking
        Baseline.boundary_check(TRAIN_DF)

        #HAI train dataset setting with stride => Define Dataset interface in pytorch
        #Stride = sliding size, 10 secs interval
        HAI_DATASET_TRAIN = HaiDataset(TRAIN_DF_RAW[TIMESTAMP_FIELD], TRAIN_DF, stride)

        #Model = bidirectional GRU
        #Hidden cell size = 100
        #Not Drop out
        #Do skip connection(RNN + first output)
        self.MODEL = StackedGRU(n_tags=TRAIN_DF.shape[1])
        #Do Cuda with torch
        self.MODEL.cuda()

        #Model training
        self.MODEL.train()
        BEST_MODEL, LOSS_HISTORY = Baseline.train(HAI_DATASET_TRAIN, self.MODEL, BATCH_SIZE, epoch)

        #Save Trained model
        with open("model.pt", "wb") as f:
            torch.save(
                {
                    "state": BEST_MODEL["state"],
                    "best_epoch": BEST_MODEL["epoch"],
                    "loss_history": LOSS_HISTORY,
                },
                f,
            )

        #Load model
        with open("model.pt", "rb") as f:
            SAVED_MODEL = torch.load(f)

        #load states
        self.MODEL.load_state_dict(SAVED_MODEL["state"])

        #Check graphs
        '''
        plt.figure(figsize=(16, 4))
        plt.title("Training Loss Graph")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.yscale("log")
        plt.plot(SAVED_MODEL["loss_history"])
        plt.show()
        '''

    def Validation(self, VALIDATION_DATASET):
        #Load Validation set
        VALIDATION_DF_RAW = Baseline.dataframe_from_csvs(VALIDATION_DATASET)

        #Normalization validation set
        VALIDATION_DF = Baseline.normalize(VALIDATION_DF_RAW[self.VALID_COLUMNS_IN_TRAIN_DATASET], self.TAG_MIN, self.TAG_MAX)

        #Validation checks boundary
        Baseline.boundary_check(VALIDATION_DF)

        self.HAI_DATASET_VALIDATION = HaiDataset(
            VALIDATION_DF_RAW[TIMESTAMP_FIELD], VALIDATION_DF, attacks=VALIDATION_DF_RAW[ATTACK_FIELD]
        )

        #Evaluate models
        self.MODEL.eval()
        CHECK_TS, CHECK_DIST, CHECK_ATT = Baseline.inference(self.HAI_DATASET_VALIDATION, self.MODEL, BATCH_SIZE)

        #Get Anomaly score
        ANOMALY_SCORE = np.mean(CHECK_DIST, axis=1)

        #Setting Threshold
        self.THRESHOLD = 0.029545

        #Check Graph
        Baseline.check_graph(ANOMALY_SCORE, CHECK_ATT, 2, self.THRESHOLD)

        LABELS = Baseline.put_labels(ANOMALY_SCORE, self.THRESHOLD)
        ATTACK_LABELS = Baseline.put_labels(np.array(VALIDATION_DF_RAW[ATTACK_FIELD]), threshold=0.5)
        FINAL_LABELS = Baseline.fill_blank(CHECK_TS, LABELS, np.array(VALIDATION_DF_RAW[TIMESTAMP_FIELD]))

        TaPR = etapr.evaluate_haicon(anomalies=ATTACK_LABELS, predictions=FINAL_LABELS)
        print(f"F1: {TaPR['f1']:.3f} (TaP: {TaPR['TaP']:.3f}, TaR: {TaPR['TaR']:.3f})")
        print(f"# of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
        #print(f"Detected anomalies: {TaPR['Detected_Anomalies']}")

        return TaPR['f1']

    def Testing(self, TEST_DATASET):

        #Get test set
        TEST_DF_RAW = Baseline.dataframe_from_csvs(TEST_DATASET)

        #Baseline.Normalize test
        TEST_DF = Baseline.normalize(TEST_DF_RAW[self.VALID_COLUMNS_IN_TRAIN_DATASET], self.TAG_MIN, self.TAG_MAX).ewm(alpha=0.9).mean()

        #Test set boundary checking
        Baseline.boundary_check(TEST_DF)

        HAI_DATASET_TEST = HaiDataset(
            TEST_DF_RAW[TIMESTAMP_FIELD], TEST_DF, attacks=None
        )
        #print(HAI_DATASET_VALIDATION[0])

        self.MODEL.eval()
        CHECK_TS, CHECK_DIST, CHECK_ATT = Baseline.inference(HAI_DATASET_TEST, self.MODEL, BATCH_SIZE)

        ANOMALY_SCORE = np.mean(CHECK_DIST, axis=1)

        #Check graph
        #Baseline.check_graph(ANOMALY_SCORE, CHECK_ATT, piece=3, THRESHOLD=self.THRESHOLD)

        LABELS = Baseline.put_labels(ANOMALY_SCORE, self.THRESHOLD)

        submission = pd.read_csv('235757_HAICon2021_dataset/sample_submission.csv')
        submission.index = submission['timestamp']
        submission.loc[CHECK_TS,'attack'] = LABELS
        #print(submission)

        submission.to_csv('baseline.csv', index=False)

    #Setting labeling
    def put_labels(distance, threshold):
        xs = np.zeros_like(distance)
        xs[distance > threshold] = 1
        return xs

    def fill_blank(check_ts, labels, total_ts):
        def ts_generator():
            for t in total_ts:
                yield dateutil.parser.parse(t)

        def label_generator():
            for t, label in zip(check_ts, labels):
                yield dateutil.parser.parse(t), label

        g_ts = ts_generator()
        g_label = label_generator()
        final_labels = []

        try:
            current = next(g_ts)
            ts_label, label = next(g_label)
            while True:
                if current > ts_label:
                    ts_label, label = next(g_label)
                    continue
                elif current < ts_label:
                    final_labels.append(0)
                    current = next(g_ts)
                    continue
                final_labels.append(label)
                current = next(g_ts)
                ts_label, label = next(g_label)
        except StopIteration:
            return np.array(final_labels, dtype=np.int8)

    def check_graph(xs, att, piece=2, THRESHOLD=None):
        l = xs.shape[0]
        chunk = l // piece
        fig, axs = plt.subplots(piece, figsize=(20, 4 * piece))
        for i in range(piece):
            L = i * chunk
            R = min(L + chunk, l)
            xticks = range(L, R)
            axs[i].plot(xticks, xs[L:R])
            if len(xs[L:R]) > 0:
                peak = max(xs[L:R])
                axs[i].plot(xticks, att[L:R] * peak * 0.3)
            if THRESHOLD!=None:
                axs[i].axhline(y=THRESHOLD, color='r')
        plt.show()

#Do train
    def train(dataset, model, batch_size, n_epochs):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters())
        loss_fn = torch.nn.MSELoss()
        epochs = trange(n_epochs, desc="training")
        best = {"loss": sys.float_info.max}
        loss_history = []
        for e in epochs:
            epoch_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                given = batch["given"].cuda()
                guess = model(given)
                answer = batch["answer"].cuda()
                loss = loss_fn(answer, guess)
                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()
            loss_history.append(epoch_loss)
            epochs.set_postfix_str(f"loss: {epoch_loss:.6f}")
            if epoch_loss < best["loss"]:
                best["state"] = model.state_dict()
                best["loss"] = epoch_loss
                best["epoch"] = e + 1
        return best, loss_history

    def inference(dataset, model, batch_size):
        dataloader = DataLoader(dataset, batch_size=batch_size)
        ts, dist, att = [], [], []
        with torch.no_grad():
            for batch in dataloader:
                given = batch["given"].cuda()
                answer = batch["answer"].cuda()
                guess = model(given)
                ts.append(np.array(batch["ts"]))
                dist.append(torch.abs(answer - guess).cpu().numpy())
                try:
                    att.append(np.array(batch["attack"]))
                except:
                    att.append(np.zeros(batch_size))
                
        return (
            np.concatenate(ts),
            np.concatenate(dist),
            np.concatenate(att),
        )

    def dataframe_from_csv(target):
        return pd.read_csv(target).rename(columns=lambda x: x.strip())

    def dataframe_from_csvs(targets):
        return pd.concat([Baseline.dataframe_from_csv(x) for x in targets])

    def boundary_check(df):
        x = np.array(df, dtype=np.float32)
        return np.any(x > 1.0), np.any(x < 0), np.any(np.isnan(x))

    #Normalization
    def normalize(df, TAG_MIN, TAG_MAX):
        ndf = df.copy()
        for c in df.columns:
            if TAG_MIN[c] == TAG_MAX[c]:
                ndf[c] = df[c] - TAG_MIN[c]
            else:
                ndf[c] = (df[c] - TAG_MIN[c]) / (TAG_MAX[c] - TAG_MIN[c])
        return ndf

    def Doin(self):
        #Do training
        self.Training(TRAIN_DATASET, stride, epoch)
        #Do Validation
        f1_score = self.Validation(VALIDATION_DATASET)
        #Do Testing
        self.Testing(TEST_DATASET)

        return f1_score

    def __init__(self):
        print("[+] Baseline Start")


if __name__ == "__main__":
    Baseline().Doin()