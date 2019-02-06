import pickle
import pandas as pd
import torch

subjects = ["subj10"]


directory = "data/grasp-and-lift-eeg-detection/train/"

input_tensors = []
events_tensors = []

for s in subjects:
    sub_data = pd.read_csv(directory + s + "_series1_data.csv", sep=",")

    sub_events = pd.read_csv(directory + s + "_series1_events.csv", sep=",")
    # TODO: z-score inputs
    input_tensor = torch.Tensor(sub_data.drop("id", axis=1).values)\
                        .view(1, sub_data.shape[0], sub_data.shape[1] - 1)
    events_tensor = torch.Tensor(sub_events.drop("id", axis=1).values)\
                         .view(1, sub_events.shape[0], sub_events.shape[1] - 1)

    input_tensors.append(input_tensor)
    events_tensors.append(events_tensor)

input_tensors = torch.cat(input_tensors)
events_tensors = torch.cat(events_tensors)

with open(f"data/tensors.pkl", "wb") as f:
    pickle.dump((input_tensors, events_tensors), f)
