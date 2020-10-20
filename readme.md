# Research Replication Prediction Using Weakly Supervised Learning

We propose two weakly supervised learning approaches that use automatically extracted text information of research papers to improve the prediction accuracy of research replication using both labeled and unlabeled datasets. 

## Experimental settings

This project is tested under the following environment settings:
- OS: Ubuntu 16.04 LTS
- GPU: Geforce 1080 Ti Cuda: 9.0, cuDNN: 7.0
- Python: 3.6
- PFMiner: pdfminer3k-1.3.4

## Datasets

We will upload it later.

## Codes

The code of crawling the unsupervised corpus: crawler_papers.py.

The code of our proposed variational inference aided weakly supervised learning method: research_replication_prediction_variational_inference.py.

The code of our proposed peer loss aided weakly supervised learning method: research_replication_prediction_peer_loss.py. The model which has 71.72% accuracy using text-only features is as follows: [71.72% model using text-only features for Peer Loss aided Weakly Supervised Learning](https://drive.google.com/file/d/1EMWTaC0KQHBwupVQ03d7VTpwQXwK83Ca/view?usp=sharing).

The code of statistics model using SVM method: svm_statistics.py.

Our BERT pretrained model files using our own corpus are in the following google drive link:
[BERT pretrained model files using our own corpus](https://drive.google.com/file/d/1Wu_hp2OWe9y0Zwt9h2PdBDO6dzDvhzue/view?usp=sharing).
