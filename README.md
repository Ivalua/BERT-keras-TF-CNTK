# BERT-keras-TF-CNTK

Please find our BERT model, tested for CNTK and Tensorflow.

`--init_checkpoint` enables function do load weights from Tensorflow checkpoint (under Tensorflow only). Otherwise, weights in Keras h5 format can be downloaded [here](https://s3-eu-west-1.amazonaws.com/christopherbourez/public/2019-01-22_13-03-29_d65cd46_bert.h5)

Tested with:
- Python 3.5.2
- Keras 2.1.5
- Tensorflow 1.5.0
- CNTK 2.6
