import pickle
name=input("enter")
with open(name, 'rb') as f:
    model = pickle.load(f)
print(model.summary())