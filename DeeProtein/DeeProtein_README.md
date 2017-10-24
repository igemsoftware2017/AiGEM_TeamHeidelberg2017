## Prerequisities

- Python 3.5+
- [Tensorflow 1.2+](https://www.tensorflow.org/api_docs/)
- [Tensorlayer 1.5.4+](http://tensorlayer.readthedocs.io/en/latest/)
- [GO-Tools](https://github.com/tanghaibao/goatools)
- [SciPy](http://www.scipy.org/install.html)

## Description

This package contains a Python library with the following funcitonality:

- a comprehensive datapreprocessing pipeline, from the uniprot/swissprot download to the final train and valid datasets.
- a pretrained but extendable deep residual neural network to classify protein sequences for GO-terms:
  The model was trained on uniport database and achieved after 13 epochs an AUC under the ROC of 99%.
  [ROC](http://2017.igem.org/wiki/images/8/89/T--Heidelberg--2017_DP_ROC.png)
  and with an average F1 score of 78%:
  [PR](http://2017.igem.org/wiki/images/f/f4/T--Heidelberg--2017_DP_Precision.png)
- a pretrained protein embedding generator, applicable for the embedding of protein sequences.


## Usage

1. Clone this git repository:
   ```bash
   $ git clone [/]
   ````
   
2. To infer a sequence on the pretrained model:
   ```bash
   $ python infer.py --gpu=True 'MSGDRETCSTGLKFJI...'
   ````
   
3. For training a custom model:
   ```bash
   $ wget ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz
   ````
   ```bash
   $ python DatasetGenerator.py --sp_load 'path_to_download'
   ````
   After preprocessing is done edit the `config_dict` and call:
   ```bash
   $ python train.py --config config_dict.JSON
   ````
   
## Documentation

Please find the [full code documentation](http://2017.igem.org/Team:Heidelberg/Software) on the iGEM wiki page of the 2017 
Heidelberg team.

