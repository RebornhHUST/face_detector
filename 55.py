import pickle

labels = {}
with open("label.pickle","rb") as f:
    labels = pickle.load(f)

