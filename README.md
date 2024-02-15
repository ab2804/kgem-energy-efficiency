# Analysis of the energy efficiency of knowledge graph embedding models

The aim of this project is to analyze the energy efficiency of knowledge graph embedding models. Not only the training
of such models with [PyKEEN](https://github.com/pykeen/pykeen) is considered, but also the application of the models for 
link prediction. The power consumption and runtime are measured using the python package 
[CodeCarbon](https://github.com/mlco2/codecarbon). The model size, the number of parameters and various quality metrics 
are also determined. The efficiency is evaluated using the framework
[STREP](https://github.com/raphischer/strep).

## Repository overview
The models are trained and used for link prediction with 'run.py'. Various metrics are determined to assess energy 
efficiency at the same time. The measured values are stored in a database with 'parse_logs.py'. By executing 'main.py',
the results are displayed and visualised (e.g. as scatter diagrams and energy labels). Further graphics are generated
with 'thesis_plots.py', which can also be found in the 'thesis_plots'-folder. In addition to the 'kgem'-database from
this project, the 'database'-folder also contains other databases for other machine learning methods for comparison
(from STREP).