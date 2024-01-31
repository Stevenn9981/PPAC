## PPAC

This is the source code of PPAC.

Paper: Debiasing Recommendation with Popular Popularity (WWW'24)

### Running environment

```
conda create -n ppac python=3.8.8
conda activate ppac

pip install torch==1.13.1 --index-url https://download.pytorch.org/whl/cu116
pip install gym==0.23.0 tensorflow-probability==0.20.1 matplotlib scikit-learn
pip install  dgl==1.1.3 -f https://data.dgl.ai/wheels/cu116/repo.html

```

### Guidelines to run our codes

If you want to train PPAC, when you use mf/ncf as base model, please use the following scripts:

```
python run_MF.py --model ppacmf --dataset {dataset} --train
python run_MF.py --model ppacncf --dataset {dataset} --train
```

If you want to use LightGCN as base model and train PPAC:

`
python run_LightGCN.py --model ppaclg --dataset {dataset} --train
`

${dataset} can be chosen from ['ml-1M', 'gowalla', 'yelp2018'].

After training, if you want to use your pre-trained model to conduct inference, use the below script (remove `--train` flag).

```
python run_MF.py --model ppacmf --dataset {dataset} --gamma {gamma} --beta {beta}
python run_MF.py --model ppacncf --dataset {dataset} --gamma {gamma} --beta {beta}
python run_LightGCN.py --model ppaclg --dataset {dataset} --gamma {gamma} --beta {beta}
```


We provide the hyper-parameters used in our experiments for your references.

|          | ML-1M    |         | Gowalla  |         | Yelp2018 |         |
|----------|----------|---------|----------|---------|----------|---------|
|          | $\gamma$ | $\beta$ | $\gamma$ | $\beta$ | $\gamma$ | $\beta$ |
| BPRMF    | 64       | -32     | 512      | -1024   | 256      | -512    |
| NCF      | 32       | -16     | 64       | -256    | 128      | -256    |
| LightGCN | 16       | -8      | 64       | -512    | 32       | -128    |




For example:

```
python run_MF.py --model ppacmf --dataset gowalla --gamma 512 --beta -1024
```

If you use our datasets or codes, please cite our paper.
```
@inproceedings{PPAC,
    author = {Ning, Wentao and Cheng, Reynold and Yan, Xiao and Kao, Ben and Huo, Nan and Haldar, Nur Al Hasan and Tang, Bo},
    title = {Debiasing Recommendation with Popular Popularity},
    booktitle = {WWW},
    publisher = {ACM},
    year = {2024}
}
```
