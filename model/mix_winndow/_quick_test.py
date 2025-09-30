import numpy as np, subprocess, sys, os, json, pathlib, tempfile
root = pathlib.Path(__file__).parent
N=200
short = np.random.rand(N).astype('float32')
long = np.clip(short*0.6 + np.random.rand(N).astype('float32')*0.4, 0,1)
labels = ( (short*0.4 + long*0.6) > 0.55 ).astype('int64')
np.save(root/'short_probs.npy', short)
np.save(root/'long_probs.npy', long)
np.save(root/'labels.npy', labels)
cmd = [sys.executable, str(root/'train_mix.py'), '--short-probs', str(root/'short_probs.npy'), '--long-probs', str(root/'long_probs.npy'), '--labels', str(root/'labels.npy'), '--fusion','mlp','--epochs','1','--val-ratio','0.2','--out', str(root/'_out_test'), '--sweep-thresholds']
print('Running:', ' '.join(cmd))
subprocess.run(cmd, check=True)
print('Done. List out dir:')
print(os.listdir(root/'_out_test'))
