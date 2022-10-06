# Code for LD-MAN

## Prerequisites

* Python 3.5
* Pytorch ≥ 1.0
* CUDA 10.0


## Installation

1. Download the checkpoints and preprocessed data: [ckpt](https://pan.baidu.com/s/1Tti39gAHnjYT0OrF6deAhw)(Extraction code: ```uoae```); [data](https://pan.baidu.com/s/1phGdJTGiMdexYNaljjm1gA)(Extraction code: ```2qmc```); [glove](https://pan.baidu.com/s/1EONiX9YV8CmunMKikPBLTA)(Extraction code: ```h1e4```)
2. Put the files in corresponding folders.


## Evaluation

Test LD-MAN with:

```bash
python release_main.py --dataset ${DATASET}
```

where ```DATASET={DMON, RON}```

## Results

Since the original code is not available now, we reproduce the LD-MAN. The results are:

|  | Reported | Reproduced |
| :-----: | :----: | :----: |
| RON | 53.51 | 53.03 |
| DMON | 80.81 | 81.15 |


