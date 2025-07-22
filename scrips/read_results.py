import numpy as np
with open("/data/zzx/DPVO_E2E/scrips/aug_dpvo_result.txt", 'r') as f:
    lines = f.readlines()
easy_ate = [float(k.strip().split()[1]) for k in lines if 'Easy' in k]
hard_ate = [float(k.strip().split()[1]) for k in lines if 'Hard' in k]
print("Easy ATE:", np.mean(easy_ate))
print("Hard ATE:", np.mean(hard_ate))
print("Overall ATE:", np.mean(easy_ate+hard_ate))
print("AUC:",np.maximum(1 - np.array(easy_ate+hard_ate), 0).mean())