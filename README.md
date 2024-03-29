## Cascading Failure Learning

The relevant data and codes of our ISCAS2022 work "**Predicting Onset Time of Cascading Failure in Power Systems Using A Neural Network-Based Classifier**". [[Paper]](https://ieeexplore.ieee.org/abstract/document/9937800)

<p align="center"> <img src="fig/framework.png" /> <p align="center"><em>Fig. 1. A systematic framework of the application of neural network-based classifier for predicting the degree of urgency of the onset time that is identified from failure propagation in a power system.</em></p>



## Requirements

- torch
- networkx
- numpy
- sklearn
- scipy # for loading matlab matrix



## Data description

In the **/data** folder, we provide the following samples for training or testing.

- Nm2: All samples (i.e., power parameter matrices and corresponding labels) on N-2 security criteria.
- Nm2_changed: 5000 samples generated from changing original state on N-2 security criteria.
- Nm3 to Nm5: Corresponding samples generated on N-3/4/5 security criteria, 5000 samples for each subset.



## Code description

- main_mlp_topo.py: implements the mlp-based prediction task.
- utils.py: implements some basic functions such as data loading, label transform, etc.



## Run the demo

```
python main_mlp_topo.py
```



## Cite

If you find this work is helpful, please cite our paper. Thank you.

```
@inproceedings{fang2022predicting,
  title={Predicting onset time of cascading failure in power systems using a neural network-based classifier},
  author={Fang, Junyuan and Liu, Dong and Tse, Chi, K},
  booktitle={2022 IEEE International Symposium on Circuits and Systems (ISCAS)},
  pages={3522--3526},
  year={2022},
  organization={IEEE}
}
```

