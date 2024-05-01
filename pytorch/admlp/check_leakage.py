import pickle
import pandas as pd
from tqdm import tqdm

pd.set_option('display.max_columns', None)
pd.set_option("display.width", 0)

def make_getters():
    getters = []

    for v in range(3):
        getters.append(lambda x: x['v'][v])
    
    for a in range(3):
        getters.append(lambda x: x['a'][v])

    for x1 in range(4):
        for x2 in range(3):
            getters.append(lambda x: x['x'][x1][x2])

    for gt1 in range(7):
        for gt2 in range(3):
            getters.append(lambda x: x['gt'][gt1][gt2])
    return getters

def check_triplet(a, b):
    

    if a[0] == b[0] and a[1] == b[1] and a[2] == b[2]:
        return True
    
    return False


def cross_check(entry):
    checked = 0
    same = 0
    for gt1 in range(len(entry['gt'])):
        gt = entry['gt'][gt1]
        checked += 1
        if check_triplet(gt, entry['a']):
            print(f"Gt{gt1} and acceleration equal")
            same += 1
        checked += 1
        if check_triplet(gt, entry['v']):
            print(f"Gt{gt1} and velocity equal")
            same += 1

        for x in range(4):
            checked += 1
            if check_triplet(gt, entry['x'][x]):
                print(f"Gt{gt1} and x{x} equal")   
                same += 1
    return checked, same


def preprocess(data):
    keys = dict(data).keys()

    columns = []
    for i in range(3):
        columns.append(f"a_{i}")
        columns.append(f"v_{i}")
        for j in range(4):
            columns.append(f"x_{j}_{i}") 
        for j in range(7):
            columns.append(f"gt_{j}_{i}")
    
    out = []

    for key in keys:
        entry = data[key]
        d = dict()
        for i in range(3):
            d[f"a_{i}"] = entry['a'][i]
            d[f"v_{i}"] = entry['v'][i]
            for j in range(4):
                d[f"x_{j}_{i}"] = entry['x'][j][i]
            for j in range(len(entry['gt'])):
                d[f"gt_{j}_{i}"] = entry['gt'][j][i]
        out.append(d)
    
    out = pd.DataFrame(out)
    return out

def check_leakage(data):
    keys = dict(data).keys() 
    
    total_checked = 0
    total_same = 0

    for key in keys:
        checked, same = cross_check(data[key])

        total_checked += checked
        total_same += same
    print(f"Total checked: {total_checked}\nTotal same: {total_same}\nRatio:{total_same/total_checked}")

def load_nuscene():
    with open("./stp3_val/data_nuscene.pkl", 'rb') as f:
        data = pickle.load(f)
    return data



def load_carla():
    with open("./data.pkl", "rb") as f:
        data = list(pickle.load(f))
        data = {k: v for k, v in enumerate(data)}
    return data

if __name__ == "__main__":
    data = load_carla()
    df = preprocess(data)
    with open("carla_corr.txt", "w") as f:
        f.write(str(df.corr()))
    #check_leakage(data)
