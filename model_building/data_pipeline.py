import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import html
import re
import string

ACCENT_CHARS = ['À', 'Á', 'Â', 'Ã', 'Ä', 'Å', 'à', 'á', 'â', 'ã', 'ä', 'å', 'È', 'É', 'Ê', 'Ë', 'è', 'é', 'ê', 'ë',
                'Ì', 'Í', 'Î', 'Ï', 'ì', 'í', 'î', 'ï', 'Ò', 'Ó', 'Ô', 'Õ', 'Ö', 'ò', 'ó', 'ô', 'õ', 'ö', 'Ù', 'Ú',
                'Û', 'Ü', 'ù', 'ú', 'û', 'ü', 'Ý', 'ý', 'ÿ', 'Ç', 'ç', 'Ñ', 'ñ', 'Æ', 'æ', 'Œ', 'œ']

TRANSLATE_TABLE = str.maketrans({c:' ' for c in ACCENT_CHARS})


def clean_text(text):
    """
    Cleans the input text.

    Args:
        text: The string to clean

    Returns:
        The cleaned text.
    """

    # Account for html characters
    new_text = html.unescape(text)

    # Remove links
    new_text = re.sub('http.{15,19}', ' ', new_text)

    # Remove numbers, and @ tags
    new_text = re.sub('\d|@\w+|\x89', ' ', new_text)

    # Account for #'d CamelCase
    new_text = re.sub('([a-z])([A-Z])', '\\1 \\2', new_text)

    # Remove punctuation
    new_text = re.sub(f"[{re.escape(string.punctuation)}]", ' ', new_text)

    # Remove single characters
    new_text = re.sub(' \w ', ' ', new_text)
    new_text = re.sub(' \w ', ' ', new_text)

    # Remove accent characters (they appear to be mostly junk characters)
    new_text = new_text.translate(TRANSLATE_TABLE)

    # Remove extra spaces
    new_text = re.sub('\s+', ' ', new_text)
    
    return new_text


def prep_data(data_dir:str):
    """
    Preps data files for training.

    Args:
        data_dir: The main directory the data is in.
    
    Returns:
        df: DataFrame of the training labels.
    """

    # Read in labels and create a dataframe to handle file mapping
    df = pd.read_csv(f'{data_dir}/train.csv')
    df.fillna('', inplace=True)
    df['target'] = df['target'].astype(float)
    df['og_text'] = df['text']
    df['text'] = df['text'].apply(clean_text)

    return df


def get_train_val_test_data(df:pd.DataFrame, use_half_data:bool=False):
    """
    Creates the training, validation, and test data splits.

    Args:
        df: The DataFrame to split
        use_half_data: A boolean where True means that the entire dataset
            is reduced by 50%.
    
    Returns:
        (train_df, val_df, test_df): The training, validation, and test datasets
    """

    if use_half_data:
        input_df, _ = train_test_split(df, test_size=0.5, stratify=df['target'], random_state=15)
    else:
        input_df = df

    train_df, test_val_df = train_test_split(input_df, test_size=0.3, stratify=input_df['target'], random_state=15)
    val_df, test_df = train_test_split(test_val_df, test_size=0.25, stratify=test_val_df['target'], random_state=15)

    print(f'Training set: 70%, Validation set: {0.3*0.75:.1%}, Test set: {0.3*0.25:.1%}')

    return train_df, val_df, test_df


def configure_for_performance(ds, batch_size, shuffle=True):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.cache()

    #Turn off this shuffle so that images and labels could be re-mapped together
    if shuffle:
        ds = ds.shuffle(buffer_size=1000, seed=15) 
    
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def create_tensorflow_datasets(train_df:pd.DataFrame, val_df:pd.DataFrame, test_df:pd.DataFrame, batch_size:int):
    """
    Converts datasets into compatible tensorflow datasets.

    Args:
        train_df: The training dataset
        val_df: The validation dataset
        test_df: The test dataset
        batch_size: The batch size to use during training.

    Returns:
        train_ds, val_ds, test_ds: The final versions
            of the training, validation, and test datasets
    """

    train_ds = tf.data.Dataset.from_tensor_slices((train_df['text'].values, train_df['target'].values))
    val_ds = tf.data.Dataset.from_tensor_slices((val_df['text'].values, val_df['target'].values))
    test_ds = tf.data.Dataset.from_tensor_slices((test_df['text'].values, test_df['target'].values))

    AUTOTUNE = tf.data.AUTOTUNE

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_ds = configure_for_performance(train_ds, batch_size)
    val_ds = configure_for_performance(val_ds, batch_size)
    test_ds = configure_for_performance(test_ds, batch_size, shuffle=False)

    return train_ds, val_ds, test_ds


def predict_on_kaggle_test_set(kaggle_test_dir:str, model, final_activation:str):
    """
    Creates a kaggle submission file.

    Args:
        kaggle_test_dir: The directory for the kaggle test data.
        model: The tensorflow model
        final_activation: The activation function used in the final later.
            It should be either relu or sigmoid.

    Returns:
        The final submission DataFrame
    """

    test_df = pd.read_csv(f'{kaggle_test_dir}/test.csv')
    test_df.fillna('', inplace=True)
    test_df['text'] = test_df['text'].apply(clean_text)

    predictions = model.predict(test_df['text'].values)
    
    if final_activation == 'sigmoid':
        thresh = 0.5
    elif final_activation == 'relu':
        thresh = 0

    test_df['target'] = (predictions > thresh).astype(int)
    final_submission = test_df[['id', 'target']].copy()

    return final_submission