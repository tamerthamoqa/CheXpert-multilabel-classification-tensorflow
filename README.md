# CheXpert-multilabel-classification-tensorflow
__Operating System__: Ubuntu 18.04 (you may face issues importing the packages from the requirements.yml file if your OS differs).

Code repository for training multi-label classification models on the [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) [[1](#references)] Chest X-ray dataset.


### Dataset
* CheXpert dataset original size (register your email and a download link will be sent as an email): [Official CheXpert Website](https://stanfordmlgroup.github.io/competitions/chexpert/)
* CheXpert dataset resized to 512x512 with maintained aspect ratio (used in experiments): [Drive](https://drive.google.com/file/d/1ir6kGK1yhqZZK5-2W0_JMawmcNZGc6r5/view?usp=sharing)


### Training Model
```
    usage: train_multilabel_classification_model.py [-h] [--data_dir DATA_DIR]
                                                    [--model_architecture {densenet201,inceptionresnetv2,resnet152}]
                                                    [--train_multi_gpu TRAIN_MULTI_GPU]
                                                    [--num_gpus NUM_GPUS]
                                                    [--training_epochs TRAINING_EPOCHS]
                                                    [--resume_train RESUME_TRAIN]
                                                    [--optimizer {sgd,adam,nadam}]
                                                    [--lr LR]
                                                    [--use_nesterov_sgd USE_NESTEROV_SGD]
                                                    [--use_amsgrad_adam USE_AMSGRAD_ADAM]
                                                    [--batch_size BATCH_SIZE]
                                                    [--image_height IMAGE_HEIGHT]
                                                    [--image_width IMAGE_WIDTH]
                                                    [--num_workers NUM_WORKERS]
    
    Training CheXpert multi-label classification model.
    
    optional arguments:
      -h, --help            show this help message and exit
      --data_dir DATA_DIR   (Required) Path to the CheXpert dataset folder
                            (default: 'dataset/')
      --model_architecture {densenet201,inceptionresnetv2,resnet152}
                            The required model architecture for training:
                            ('densenet201','inceptionresnetv2', 'resnet152'),
                            (default: 'densenet201')
      --train_multi_gpu TRAIN_MULTI_GPU
                            If set to True, train model with multiple GPUs.
                            (default: False)
      --num_gpus NUM_GPUS   Set number of available GPUs for multi-gpu training, '
                            --train_multi_gpu' must be also set to True (default:
                            1)
      --training_epochs TRAINING_EPOCHS
                            Required training epochs (default: 15)
      --resume_train RESUME_TRAIN
                            If set to True, resume model training from model_path
                            (default: False)
      --optimizer {sgd,adam,nadam}
                            Required optimizer for training the model:
                            ('sgd','adam','nadam'), (default: 'adam')
      --lr LR               Learning rate for the optimizer (default: 0.0001)
      --use_nesterov_sgd USE_NESTEROV_SGD
                            Use Nesterov momentum with SGD optimizer: ('True',
                            'False') (default: False)
      --use_amsgrad_adam USE_AMSGRAD_ADAM
                            Use AMSGrad with adam optimizer: ('True', 'False')
                            (default: False)
      --batch_size BATCH_SIZE
                            Input batch size, if --train_multi_gpu then the
                            minimum value must be the number of GPUs (default: 16)
      --image_height IMAGE_HEIGHT
                            Input image height (default: 512)
      --image_width IMAGE_WIDTH
                            Input image width (default: 512)
      --num_workers NUM_WORKERS
                            Number of workers for fit_generator (default: 4)
```


### References:
* [1] Irvin, J.A., Rajpurkar, P., Ko, M., Yu, Y., Ciurea-Ilcus, S., Chute, C., Marklund, H., Haghgoo, B., Ball, R.L., Shpanskaya, K.S., Seekins, J., Mong, D.A., Halabi, S.S., Sandberg, J.K., Jones, R.H., Larson, D.B., Langlotz, C., Patel, B.N., Lungren, M.P., & Ng, A. (2019). CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. AAAI. [arXiv](https://arxiv.org/abs/1901.07031)


### Hardware Specifications
* TITAN RTX Graphics Card (24 gigabytes Video RAM).
* i9-9900KF Intel CPU overclocked to 5 GHz.
* 32 Gigabytes DDR4 RAM at 3200 MHz.
