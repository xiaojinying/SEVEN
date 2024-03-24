<h2 align="center">SEVEN：pruning transformer model by reserving sentinels</h2>
<p align="center"><b>IJCNN 2024</b> | <a href="https://arxiv.org/pdf/2403.12688.pdf">[Paper]</a> | <a href="https://github.com/xiaojinying/SEVEN">[Code]</a> </p>

Our results achieved

![image text](https://github.com/xiaojinying/SEVEN/fig/plot1.png)
![image text](https://github.com/xiaojinying/SEVEN/fig/plot2.png)


## TODO List

- [x] Plug-and-Play Implementation of SEVEN
- [ ] Experiment: SEVEN<sub>pre</sub> on ImageNet
- [ ] Experiment: SEVEN<sub>pre</sub> on Cifar
- [x] Experiment: SEVEN<sub>dyn</sub> on GLUE

## Contents
- [Clone](#install)
- [Requirements](#Requirements)
- [Experiments](#experiments)
- [Citation](#citation)

## Clone

You can clone this repo and install it locally.

```bash
git clone https://github.com/xiaojinying/SEVEN.git
```

## Requirements
```
datasets>=2.18.0
easydict>=1.11
transformers>=4.35.2
```
## Experiments

### GLUE

To run the SST-2 example with SEVEN, run the following:
```bash
python train.py --dataset sst2 --alpha_1 0.8 --alpha_2 0.8 --learning_rate 2e-5 --epoch 10 --batchsize 32 --pruning_algo SEVEN --target_ratio 0.6
```

## Citation
```bibtex
@article{xiao2024seven,
  title={SEVEN: Pruning Transformer Model by Reserving Sentinels},
  author={Xiao, Jinying and Li, Ping and Nie, Jie and Tang, Zhe},
  journal={arXiv preprint arXiv:2403.12688},
  year={2024}
}
```
