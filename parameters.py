
class Model_Parameters:
    pass

params_dict = {
    # Experiment Params:
    'is_training': True,  # train mode (relevant only for accelerated LSTM mode)
    'data_set': 'cat',  # datasets to train on
    'epochs': 50,  # how many times to go over the full train set (on average, since batches are drawn randomly)
    'best_only': True, # Batches between checkpoints creation and validation set evaluation. Once an epoch if None.
    'batch_size': 100,  # Minibatch size. Recommend leaving at 100.
    'accelerate_LSTM': False,  # Flag for using CuDNNLSTM layer, gpu + tf backend only
    # Loss Params:
    'optimizer': 'adam',  # adam or sgd
    'learning_rate': 0.001,
    'decay_rate': 0.9999,  # Learning rate decay per minibatch.
    'min_learning_rate': .00001,  # Minimum learning rate.
    'kl_tolerance': 0.2,  # Level of KL loss at which to stop optimizing for KL.
    'kl_weight': 0.5,  # KL weight of loss equation. Recommend 0.5 or 1.0.
    'kl_weight_start': 0.01,  # KL start weight when annealing.
    'kl_decay_rate': 0.99995,  # KL annealing decay rate per minibatch.
    'grad_clip': 1.0,  # Gradient clipping. Recommend leaving at 1.0.
    # Architecture Params:
    'z_size': 128,  # Size of latent vector z. Recommended 32, 64 or 128.
    'enc_rnn_size': 256,  # Units in encoder RNN.
    'dec_rnn_size': 512,  # Units in decoder RNN.
    'use_recurrent_dropout': True,  # Dropout with memory loss. Recommended
    'recurrent_dropout_prob': 0.9,  # Probability of recurrent dropout keep.
    'num_mixture': 20,  # Number of mixtures in Gaussian mixture model.
    # Data pre-processing Params:
    'random_scale_factor': 0.15,  # Random scaling data augmentation proportion.
    'augment_stroke_prob': 0.10  # Point dropping augmentation proportion.
}

for key in params_dict:
    setattr(Model_Parameters, key, params_dict[key])
