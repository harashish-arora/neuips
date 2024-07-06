import sys
import re

with open("/DATATWO/users/solubility/harashish/src/benchmarks/methods/unimol_method.py", "r") as f:
    code = f.read()

import multiprocessing

new_code = code.replace(
"""    sanitized = []
    failed_indices = set()
    for i, smi in enumerate(smiles_list):
        clean = _sanitize_smiles(smi)
        if clean is not None:
            sanitized.append((i, clean))
        else:
            failed_indices.add(i)""",

"""    from multiprocessing import Pool
    
    sanitized = []
    failed_indices = set()
    with Pool(8) as pool:
        clean_smiles = pool.map(_sanitize_smiles, smiles_list)
        
    for i, clean in enumerate(clean_smiles):
        if clean is not None:
            sanitized.append((i, clean))
        else:
            failed_indices.add(i)""")

with open("/DATATWO/users/solubility/harashish/src/benchmarks/methods/unimol_method.py", "w") as f:
    f.write(new_code)
