# LD-MAN: Layout-Driven Multimodal Attention Network for Online News Sentiment Recognition
## Introduction

We propose a layout-driven multimodal attention network (LD-MAN) to recognize news sentiment in an end-to-end manner. Rather than modeling text and images individually, LD-MAN uses the layout of online news to align images with the corresponding text. Specifically, it exploits a set of distance-based coefficients to model the image locations and measure the contextual relationship between images and text. LD-MAN then learns the affective representations of the articles from the aligned text and images using a multimodal attention mechanism. Considering the lack of relevant datasets in this field, we collect two multimodal online news datasets.

## Prerequisites

* Python 3.5
* Pytorch ≥ 1.0
* CUDA 10.0

## Installation

- [x] [Original dataset](https://github.com/Gyaya/LD-MAN/tree/main/dataset).
- [ ] Code (to be uploaded). 


## Citation

    @article{2020LD,
        title={LD-MAN: Layout-Driven Multimodal Attention Network for Online News Sentiment Recognition},
        author={Wenya Guo, Ying Zhang, Xiangrui Cai, Lei Meng, Jufeng Yang and Xiaojie Yuan},
        journal={IEEE Transactions on Multimedia},
        volume={PP},
        number={99},
        pages={1-1},
        year={2020},
    }
