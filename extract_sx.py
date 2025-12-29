import os
from tqdm import tqdm

dset = ["train", "test"]
for d in dset:
    for i in tqdm(range(1, 9)):
        for spk in os.listdir(os.path.join("TIMIT", "timit", d, f"dr{i}")):
            for file in os.listdir(os.path.join("TIMIT", "timit", d, f"dr{i}", spk)):
                if not file.startswith("sx"):
                    continue
                id = file.split(".")[0]
                # makedir if not exist
                os.makedirs(os.path.join("SX2", id), exist_ok=True)
                os.makedirs(os.path.join("SX2", id, f"dr{i}"), exist_ok=True)
                os.makedirs(os.path.join("SX2", id, f"dr{i}", spk), exist_ok=True)
                # copy file
                os.system(f"cp {os.path.join('TIMIT', 'timit', d, f'dr{i}', spk, file)} {os.path.join('SX2', id, f'dr{i}', spk)}")


