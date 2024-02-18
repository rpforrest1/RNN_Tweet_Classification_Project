import numpy as np
import matplotlib.pyplot as plt
import json
import copy
import os
from keras import layers
from gensim import downloader
from collections import OrderedDict
from keras import metrics
from keras.models import Sequential
import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
try:
    from .data_pipeline import create_tensorflow_datasets
except:
    from data_pipeline import create_tensorflow_datasets
from time import gmtime, time



def get_vectorizer(train_ds, max_tokens, output_sequence_length=None):
    """
    Creates the text vectorizer.

    Args:
        train_ds: The training dataset.
        max_tokens: The max number of tokens in the vocabulary.
        output_sequence_length: The output sequence length for the text vectorizer.
    
    Returns:
        The tensorflow vectorizer.
    """

    vectorizer = layers.TextVectorization(max_tokens=max_tokens,
                                            output_sequence_length=output_sequence_length)
    
    vectorizer.adapt(train_ds.map(lambda text, label: text))
    
    return vectorizer


def get_embedding_layer(embed_size, vocab_size, pretrained_model_str=None, vectorizer=None):
    """
    Creates the embedding layer.

    Args:
        embed_size: The embedding size.
        vocab_size: The vocabulary size.
        pretrained_model_str: A string that defines the gensim pretrained model. If None,
            the embedding will be trained within the model.
        vectorizer: The text vectorizer.
    
    Returns:
        The tensorflow embedding layer.
    """

    if pretrained_model_str:
        pretrained_model = downloader.load(pretrained_model_str)

        vocabulary = vectorizer.get_vocabulary()
        word_index = dict(zip(vocabulary, range(vocab_size)))
        
        converted = 0
        
        # Prepare embedding matrix
        embedding_matrix = np.zeros((vocab_size, embed_size))
        for word, i in word_index.items():
            if pretrained_model.has_index_for(word):
                embedding_vector = pretrained_model[word]
                embedding_matrix[i] = embedding_vector
                converted += 1
        
        # print(f'{converted} out of {len(word_index.items())} words converted.')

        embedding_layer = layers.Embedding(vocab_size, embed_size, mask_zero=True, trainable=False)
        embedding_layer.build((1,))
        embedding_layer.set_weights([embedding_matrix])
    
    else:
        embedding_layer = layers.Embedding(vocab_size, embed_size, mask_zero=True)

    return embedding_layer


def build_model(vectorizer, embedding_layer, use_bidirectional, rnn_layer, rnn_units, dense_units, final_dropout_rate, activation, optimizer, loss):
    """
    Builds the tensorflow model.

    Args:
        vectorizer: The tensorflow vectorizer layer object.
        embedding_layer: The tensorflow embedding layer.
        use_bidirectional: If set to True, the RNN layers will be bidirectional.
        rnn_layer: The tensorflow class object that defines the type of RNN layer to use. (layers.GRU for example)
        rnn_units: A list that defines the number of RNN layers based on the number units provided for that index.
            The value at that index is the number of units for that layer.
        dense_units: A list that defines the number of dense layers based on the number units provided for that index.
            The value at that index is the number of units for that layer.
        final_dropout_rate: A float that provides the dropout percentage after the final layer before classification.
        activation: A string for the activation function to use in all the dense layers.
        optimizer: The tensorflow class object that defines the optimizer to use.
        loss: The loss function to use. Can be a string or tensorflow object.

    Returns:
        The tensorflow model.
    """

    tf.random.set_seed(15)
    keras.utils.set_random_seed(15)

    model_metrics = OrderedDict([
        ('accuracy', metrics.BinaryAccuracy(name='accuracy')),
        ('auc', metrics.AUC(name='auc')),
        ('precision', metrics.Precision(name='precision')),
        ('recall', metrics.Recall(name='recall'))
    ])
    
    # Model Architecture
    model = Sequential()
    model.add(vectorizer)
    model.add(embedding_layer)
    
    # RNN Layers
    for i, units in enumerate(rnn_units):
        if i != (len(rnn_units)-1):
            return_sequences = True
        else:
            return_sequences = False

        if use_bidirectional:
            model.add(layers.Bidirectional(rnn_layer(units, return_sequences=return_sequences)))
        else:
            model.add(rnn_layer(units, return_sequences=return_sequences))
            
    # Dense Layers
    for i, units in enumerate(dense_units):
        if i == (len(dense_units)-1):
            # Add dropout right before the final dense layer
            if final_dropout_rate:
                model.add(layers.Dropout(final_dropout_rate))
        
        model.add(layers.Dense(units, activation=activation))
    

    # Compile Model
    model.compile(optimizer=optimizer(), loss=loss, metrics=list(model_metrics.values()))

    return model


def create_fit_and_save_model(model_name, train_df, val_df, test_df, epochs, params, final_fitting=False):
    """
    Creates the tensormodel given the parameters.

    Args:
        model_name: The string name for the model.
        train_df: The training dataframe.
        val_df: The validation dataframe.
        test_df: The test dataframe.
        epochs: The number of epochs for model training.
        params: A tuple with all model building parameters.
        final_fitting: A boolean where True means its the final batch of models to fit. This will model
            checkpoint data in a different folder name (model_checkpoints_final).

    Returns:
        The f1-score for the test dataset.
    """

    vocab_size, embed_size_with_pretrained_model, batch_size, bidirectional, rnn_layer, rnn_units, dense_units, activation, final_dropout, optimizer = params
    embed_size, pretrained_model_str = embed_size_with_pretrained_model

    train_ds, val_ds, test_ds = create_tensorflow_datasets(train_df, val_df, test_df, batch_size)
    vectorizer = get_vectorizer(train_ds, vocab_size)
    embedding_layer = get_embedding_layer(embed_size, vocab_size, pretrained_model_str=pretrained_model_str, vectorizer=vectorizer)

    model = build_model(vectorizer=vectorizer,
                        embedding_layer=embedding_layer,
                        use_bidirectional=bidirectional,
                        rnn_layer=rnn_layer,
                        rnn_units=rnn_units,
                        dense_units=dense_units,
                        activation=activation,
                        final_dropout_rate=final_dropout,
                        optimizer=optimizer,
                        loss='binary_crossentropy'
                       )
    
    if final_fitting:
        model_output_dir = './model_checkpoints_final'
    else:
        model_output_dir = './model_checkpoints'
    
    checkpoint_path = f"{model_output_dir}/{model_name}/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=0, save_best_only=True, monitor='val_loss')

    # Define parameters for early stopping and learning rate reduction
    earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=0, restore_best_weights=True)
    reducel = ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=0, factor=0.1)

    start_time = time()
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=epochs,
                        callbacks=[cp_callback, reducel, earlystopper],
                        verbose=0
                       )
    total_time = time() - start_time

    conv_time = gmtime(total_time)
    history_dict = save_history(history, model_output_dir, model_name)

    # Check test set and save results
    test_metrics = model.evaluate(test_ds, verbose=0)
    test_metrics = map_to_metrics(model.metrics_names, test_metrics)
    test_metrics['f1_score'] = 2*(test_metrics['precision']*test_metrics['recall']/(test_metrics['precision']+test_metrics['recall']+np.finfo(float).eps))
    test_metrics['total_time'] = total_time

    json.dump(test_metrics, open(f"{model_output_dir}/{model_name}/test_metrics.json", 'w'))

    return test_metrics['f1_score']


def save_history(history_obj, model_output_dir, MODEL_NAME):
    """
    Saves model history data as a JSON.

    Args:
        history_obj: The history object from the model.
        model_output_dir: The model output directory
        MODEL_NAME: The name of the model
    
    Returns:
        A dictionary of the history data.
    """

    history_dict = copy.deepcopy(history_obj.history)

    for k, v in history_dict.items():
        history_dict[k] = list(np.array(history_dict[k]).astype(float))
    
    if 'precision' in history_dict.keys() and 'recall' in history_dict.keys():
        pre = np.array(history_dict['precision'])
        rec = np.array(history_dict['recall'])
        history_dict['f1_score'] = list(2*(pre*rec/(pre+rec)))

        pre = np.array(history_dict['val_precision'])
        rec = np.array(history_dict['val_recall'])
        history_dict['val_f1_score'] = list(2*(pre*rec/(pre+rec+np.finfo(float).eps)))
        
    json.dump(history_dict, open(f"{model_output_dir}/{MODEL_NAME}/model_history.json", 'w'))

    return history_dict


def map_to_metrics(metrics_names, metric_tuple):
    """
    Maps a returned metric to its name.

    Args:
        metrics_names: The name of the metrics used for model fitting.
        metric_tuple: The tuple of model metrics
    
    Returns:
        A dictionary that maps a metric name to its value from the model.
    """

    return {key:value for key, value in zip(metrics_names, metric_tuple)}


def plot_metric(history:dict, metric_name:str, model_num=None):
    """
    Plots a model metric.

    Args:
        history: The model history dictionary containing the metrics
        metric_name: The name of the metric to plot.
        model_num: The number associated to this model.
    """
    
    label_map = {
        'loss':'Loss',
        'accuracy':'Accuracy',
        'auc':'AUC',
        'f1_score':'F1 Score'
    }
    
    plt.figure(figsize=(6,4))
    plt.plot(history[metric_name], label=metric_name)
    plt.plot(history[f'val_{metric_name}'], label=f'val_{metric_name}')
    if model_num:
        plt.title(f"Model {model_num} Training {label_map[metric_name]} vs Validation {label_map[metric_name]}")
    else:
        plt.title(f"Training {label_map[metric_name]} vs Validation {label_map[metric_name]}")
    plt.xlabel('Epoch')
    plt.ylabel(label_map[metric_name])
    plt.xticks(range(len(history[metric_name])))
    plt.legend()
    plt.show()


def plot_learning_rate(history:dict, model_num=None):
    """
    Plots the learning rate.

    Args:
        history: The model history dictionary containing the metrics
        model_num: The number associated to this model.
    """

    plt.figure(figsize=(6,4))
    plt.plot(history['lr'])
    if model_num:
        plt.title(f'Model {model_num} Learning Rate vs Epoch')
    else:
        plt.title('Learning Rate vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.xticks(range(len(history['lr'])))
    plt.show()


def bold(text):
    """
    Bold the input text for printing.

    Args:
        text: The input text.

    Returns:
        The text with the bold ANSI escape code.
    """

    return f"\033[1m{text}\033[0m"