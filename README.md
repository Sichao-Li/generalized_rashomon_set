# Exploring the Cloud of Feature Interaction Scores in a Rashomon Set


This repository includes the implementation of the paper (https://arxiv.org/abs/2305.10181). The paper emphasizes the importance of investigating feature interactions not just in a single predictive model, but across a range of well-performing models, illustrated below.

![FIS in the Rasomon set](https://github.com/Sichao-Li/generalized_rashomon_set/raw/main/data/FIn_Rset.png)

## Summary

The paper introduces a novel approach to understanding feature interactions in machine learning models. The authors argue that the study of feature interactions in a single model can miss complex relationships between features. Thus, they recommend exploring these interactions across a set of similarly accurate models, known as a Rashomon set.

The main contributions of this paper are:

1. Introduction of the Feature Interaction Score (FIS) as a means to quantify the strength of feature interactions within a predictive model.
2. Proposal of the FIS Cloud (FISC), a collection of FISs across a Rashomon set, to explore how feature interactions contribute to predictions and how these interactions can vary across models.
3. Presentation of a non-linear example in a multilayer perceptron (MLP) to characterize the Rashomon set and FISC. This includes a search algorithm to explore FIS and to characterize FISC in a Rashomon set along with two novel visualization tools, namely Halo and Swarm plots.

----

## Project structure
The project is constructed following the below structure:
```
project
│   README.md
│   requirement.txt    
│   LICENSE
└───data
│   │   data_file.csv
└───grs
│   │───explainers
│   │   │    __init__.py
│   │   │    _explainer.py
│   │───plots
│   │   │    __init__.py
│   │   │    _swarm_plot.py
│   │   │    _halo_plot.py
│   │───utils
│   │   │    __init__.py
│   │   │    _general_utils.py
│   │   __init__.py
│   │   config.py
└───demo
│   │   demo.ipynb
└───experiments
│   │   ...
└───logs
│   │   ...
└───results
│   │   ...
───────────
```

## Requirements
FISC is a designed based on Python language and specific libraries. Four most commonly-used are listed as follows:

* Python
* Jupyter Notebook
* Numpy
* Matplotlib

FISC is a model-agnostic framework and experiments might have different requirements, depending on the data and model
types. To implement the experiments in the paper, dependencies in [requirements.txt](.\requirements.txt) are required.

```python
python -r .\requirements
```

----

## Installment

```
pip install -i https://test.pypi.org/simple/ generalized-rashomon-set
```
### Usage
```
import grs

# train a MLP model
model =  MLPRegressor(hidden_layer_sizes, max_iter=200).fit(X, y)

# explain the model by exploring the Rashomon set
explainer = explainers.fis_explainer(model, X_test, y_test, epsilon_rate=0.05,
                                     loss_fn='mean_squared_error', n_ways=2, delta=0.5, wrapper_for_torch=False)

# visualize the predictions
plots.swarm_plot()
plots.halo_plot()
```
----

## Demo

A toy example shows how halo plot illustrates the effect of feature interaction 
in [toy-example.ipynb](.\demo\toy_example.ipynb), results are shown below.
![Toy example](https://github.com/Sichao-Li/generalized_rashomon_set/raw/main/demo/x1x2_simple.png
)
----

## Experiment

A more detailed example of recidivism prediction for real-world applications can be found in [experiments](.\experiments) recidivism prediction.ipynb.

The results might be slightly different from the paper due to the refactor of the code, but the main conclusions remain the same.
The package is still under refactoring to facilitate easy visualization, and we are actively working on a more comprehensive documentation.

Please feel free to drop an email to [Sichao Li](mailto:sichao.li@anu.edu.au) in case you want to discuss.
## Citation

```
@article{li2023exploring,
  title={Exploring the cloud of feature interaction scores in a Rashomon set},
  author={Li, Sichao and Wang, Rong and Deng, Quanling and Barnard, Amanda},
  journal={arXiv preprint arXiv:2305.10181},
  year={2023}
}
```




