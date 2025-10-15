import pickle 
import os 

path = "/home/or0007/gitlab/Ponita_SLR/datasets/isr/h2s/h2s_pseudo_gloss_base_with_pos_sent_test.h2s_pkl"
data = pickle.load(open(path, 'rb'))

with open('/home/or0007/gitlab/beyondbleu/out/features/H2S/h2s_test_features.pkl', 'rb') as file:
    feat_val = pickle.load(file)

print(feat_val['g0fgci8L_rc_3-8-rgb_front'].shape)