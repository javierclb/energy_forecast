import tensorflow.keras.callbacks as cl
import os 

def get_callbacks_dict(model_params):
    """ create a dictionary of all used callbacks """

    # Callbacks dictionary
    callbacks_dict = {}

    # Checkpoints callback
    callbacks_dict["model_checkpoint"] = cl.ModelCheckpoint(model_params.checkpoint , 
                                monitor='val_loss',
                                verbose = 1, 
                                save_best_only = model_params.save_best_only, 
                                mode ='min')

    # LR decay callback, modified to apply decay each batch as in original implementation
    callbacks_dict['lr_schedule'] = cl.LearningRateScheduler(
        lambda step: ((model_params.learning_rate - model_params.min_learning_rate) * model_params.decay_rate ** step
                      + model_params.min_learning_rate))

    # Tensorboard callback
    callbacks_dict["tensorboard"] = cl.TensorBoard(log_dir=os.path.join(model_params.tensorboard), histogram_freq=1)

    return callbacks_dict