# ESPDiff

ESPDiff: diffusion-driven salient object detection for oriented remote sensing imagery (ORSI-SOD), guided by edge and skeleton priors.

## Project Structure

```
.
├── train.py                    # training entry point
├── sample.py                   # inference / saliency map generation
├── eval_all_t5.py              # batch metric evaluation
├── data_aug.m                  # MATLAB data augmentation
├── config/                     # dataset path config
├── configs/                    # model / training hyperparameters
├── dataset/                    # dataset & augmentation code
├── model/                      # backbone, diffusion model, decoder, losses
│   └── net_mobile.py           # main ESPDiff network
├── denoising_diffusion_pytorch/  # vendored diffusion primitives
├── utils/                      # trainer, metrics, config, collate utils
├── results/                    # saliency maps for the three benchmarks
├── environment.yaml            # conda environment
└── requirements.txt            # pip dependencies
```

## Results

Saliency maps for the three benchmarks are provided under [`results/`](results/):

| Dataset | Path |
|---|---|
| ORSSD | `results/ORSSD/` |
| EORSSD | `results/EORSSD/` |
| ORSI-4199 | `results/ORS4199/` |
