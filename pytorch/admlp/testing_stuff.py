import pickle

data = open('fengze_nuscenes_infos_train.pkl','rb')
d = pickle.load(data)

print(d['infos'][0].keys())
print(d['infos'][0]['gt_velocity'])
