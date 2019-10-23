# MLP-NASBench-101
this implements a search agent that uses MLP to predict the top candidates for searching over NASBench-101. The architecture in NASBench-101 is encoded into a vector of length 49, consisting of flattened adjacent matrix and node list.

No GPU required, simply unzip nasbench_dataset.zip, and run python mlp.py. The final result will be written into result.txt, and each line tells the step whenever the current best accuracy improves. The search will automatically terminate once it hits the global optimum, which is found by sweeping the dataset in the beginning. Enjoy! ;) 

You can simply change the MLP architecture around line 28. Here, I'm using a fc layer that maps the networks to the accuracy. Feel free to add more additional layers.

requirements:
```
conda install -c anaconda scikit-learn
conda install pytorch torchvision -c pytorch
```
