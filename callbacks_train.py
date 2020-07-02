from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard

def get_callbacks_dict(model_params, experiment_path=''):
    """ create a dictionary of all used callbacks """

    # Callbacks dictionary
    callbacks_dict = {}

    # Checkpoints callback
    callbacks_dict["model_checkpoint"] = ModelCheckpoint('checkpoints/Weights.hdf5' , 
                                monitor='val_loss',
                                verbose = 1, 
                                save_best_only = True, 
                                mode ='min')

    # LR decay callback, modified to apply decay each batch as in original implementation
    callbacks_dict['lr_schedule'] = LearningRateSchedulerPerBatch(
        lambda step: ((model_params.learning_rate - model_params.min_learning_rate) * model_params.decay_rate ** step
                      + model_params.min_learning_rate))

    # KL loss weight decay callback, custom callback
    callbacks_dict['kl_weight_schedule'] = KLWeightScheduler(schedule=lambda step:
                                       (model_params.kl_weight - (model_params.kl_weight - model_params.kl_weight_start)
                                       * model_params.kl_decay_rate ** step), verbose=1)

    # Tensorboard callback
    callback_dict["tensorboard"] = TensorBoard(log_dir=os.path.join('tensorboard'), histogram_freq=1)

    return callbacks_dict