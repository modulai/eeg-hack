import pickle
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn_utils


def gen_list_of_series(subjects=12, series=8):
    s = []
    for a in range(1, subjects):
        for b in range(1, series):
            s.append("subj" + str(a) + "_series" + str(b))
    return s


subjects = gen_list_of_series()

directory = "data/grasp-and-lift-eeg-detection/train/"

input_tensors = []
events_tensors = []
lengths = []

for s in subjects:
    sub_data = pd.read_csv(directory + s + "_data.csv", sep=",")

    sub_events = pd.read_csv(directory + s + "_events.csv", sep=",")
    # TODO: z-score inputs
    input_tensor = torch.Tensor(sub_data.drop("id", axis=1).values)
    events_tensor = torch.Tensor(sub_events.drop("id", axis=1).values)
    lengths.append(input_tensor.shape[0])
    input_tensors.append(input_tensor)
    events_tensors.append(events_tensor)

# padd sequences
input_tensors = rnn_utils.pad_sequence(input_tensors, batch_first=True)
events_tensors = rnn_utils.pad_sequence(events_tensors, batch_first=True)
lengths = torch.Tensor(lengths)

with open(f"data/tensors.pkl", "wb") as f:
    pickle.dump((input_tensors, events_tensors, lengths), f)
