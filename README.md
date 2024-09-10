# PMT



## Installation

- Ubuntu 20.04
- PyTorch 1.11.0
- medpy, tqdm, h5py, etc.

## Getting Started

```bash
# after install dependcies
git clone git@github.com:Axi404/PMT.git
cd PMT/code
python train_PMT.py
python test_LA.py
```

## Dataset

[LA](https://drive.google.com/file/d/1Lb9Wt1TTFQ_dAaCTqeRbRI-t7S0_ZXbP/view?usp=sharing)

[Pancreas](https://drive.google.com/file/d/122JtFL_OHd9j7xdFE5V6OeikJtE-xj1D/view?usp=sharing)

Following the setting of [MCF](https://github.com/WYC-321/MCF).

## Acknowledgement

We build the project based on UA-MT, SASSNet, DTC, MCF. Thanks for their contribution.

# Citation

If you find this project useful, please consider citing:

```bibtex
@article{gao2024pmtprogressivemeanteacher,
      title={PMT: Progressive Mean Teacher via Exploring Temporal Consistency for Semi-Supervised Medical Image Segmentation}, 
      author={Ning Gao and Sanping Zhou and Le Wang and Nanning Zheng},
      year={2024},
      eprint={2409.05122},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.05122}, 
}
```