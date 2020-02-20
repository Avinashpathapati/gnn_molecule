# OMDB-GAP1
Version 1.1

| Filename       | Description                                                                                              |
|----------------|----------------------------------------------------------------------------------------------------------|
| structures.xyz | 12500 crystal structures. Use the first 10000 as training examples and the remaining 2500 as test set.   |
| bandgaps.csv   | 12500 DFT band gaps corresponding to structures.xyz                                                      |
| CODids.csv     | 12500 COD ids cross referencing the Crystallographic Open Database (in the same order as structures.xyz) |

Please cite the paper introducing this dataset: https://arxiv.org/abs/1810.12814

## Example

Example how to load the files:

```python
from ase.io import read
materials = read('structures.xyz', index=':')

import numpy as np
bandgaps = np.loadtxt('bandgaps.csv')
cods = np.loadtxt('CODids.csv', dtype=int)
```
