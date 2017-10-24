[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1035806.svg)](https://doi.org/10.5281/zenodo.1035806)

## Prerequisities

- Python 3.5+
- [Tensorflow 1.2+](https://www.tensorflow.org/api_docs/)
- [Tensorlayer 1.5.4+](http://tensorlayer.readthedocs.io/en/latest/)
- [GO-Tools](https://github.com/tanghaibao/goatools)
- [SciPy 0.19](http://www.scipy.org/install.html)
- [Pandas 0.20](https://pandas.pydata.org/pandas-docs/stable/index.html)
- [scikit-learn 0.19](http://scikit-learn.org/stable/install.html#)
- [seaborn 0.8](https://seaborn.pydata.org/index.html)


## Description

This package contains a Python library with the following funcitonality:

- a comprehensive datapreprocessing pipeline, from the uniprot/swissprot download to the final train and valid datasets.
- a pretrained protein embedding generator, applicable for the embedding of protein sequences.

- a pretrained but extendable deep residual neural network to classify protein sequences for GO-terms:
  The model was trained on uniport database and achieved after 13 epochs an AUC under the ROC of 99%.

[ROC](http://2017.igem.org/wiki/images/8/89/T--Heidelberg--2017_DP_ROC.png)

  and with an average F1 score of 78%:

[PR](http://2017.igem.org/wiki/images/f/f4/T--Heidelberg--2017_DP_Precision.png)


## Usage

1. Clone this git repository:
   ```bash
   $ git clone <link to this repo> && cd Heidelberg_2017/DeeProtein
   ````
   
2. To infer a sequence on the pretrained model:
   ```bash
   $ python infer.py --sequence=MSGDRETCSTGLKFJI...
   ````
   
3. For training a custom model:
   ```bash
   $ wget ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz
   ````
   ```bash
   $ python process_data.py --uniprot_path=path_to_download --uniprot_csv=path_to_csv --save_dir=.
   ````
   After preprocessing is done edit the `config_dict` and call:
   ```bash
   $ python train.py --config_json=config_dict.JSON
   ````
   
## Documentation

Please find the [full code documentation](http://2017.igem.org/Team:Heidelberg/Software) on the iGEM wiki page of the 2017 
Heidelberg team.

