import pandas as pd
import click
from pathlib import Path
# from typing import List
import re
import emoji
from tqdm import tqdm


def read_toxic_data(folder_toxic: str) -> pd.DataFrame:
    train_path = 'train.csv.zip'
    test_labels_path = 'test_labels.csv.zip'
    test_path = 'test.csv.zip'

    folder_toxic = Path(folder_toxic)
    train_toxic = pd.read_csv(folder_toxic.joinpath(train_path))
    print('Shape of train data:', train_toxic.shape)

    test_toxic = pd.read_csv(folder_toxic.joinpath(test_path))
    test_labels = pd.read_csv(folder_toxic.joinpath(test_labels_path))
    test_toxic = test_toxic.merge(test_labels, how='outer', left_on='id', right_on='id')
    print('Shape of test data:', test_toxic.shape)

    # -1 - unlabeled data
    test_toxic = test_toxic[test_toxic['toxic'] != -1]
    total = pd.concat([train_toxic, test_toxic])
    return total


def read_unintended_data_dense(folder_unintended: str) -> pd.DataFrame:
    folder_unintended = Path(folder_unintended)
    df = pd.read_csv(folder_unintended.joinpath('all_data.csv'))
    cols = ['id', 'comment_text', 'toxicity', 'severe_toxicity', 'obscene',
            'identity_attack', 'sexual_explicit', 'insult', 'threat']
    df['severe_toxicity'] *= 2
    df = df[cols]

    cols_with_scores = df.drop(['id', 'comment_text'], axis=1).columns
    df['offensiveness_score'] = df[cols_with_scores].mean(axis=1)

    df = df[['id', 'comment_text', 'offensiveness_score']]
    print('Shape of unintended data:', df.shape)
    return df


def read_unintended_data_sparse(folder_unintended: str, threshold: float) -> pd.DataFrame:
    folder_unintended = Path(folder_unintended)
    df = pd.read_csv(folder_unintended.joinpath('all_data.csv'))
    cols = ['id', 'comment_text', 'toxicity', 'severe_toxicity', 'obscene',
            'identity_attack', 'insult', 'threat']
    df = df[cols]
    df = df.rename({
        'toxicity': 'toxic',
        'severe_toxicity': 'severe_toxic',
        'identity_attack': 'identity_hate'},
        axis=1)

    cols_with_scores = df.drop(['id', 'comment_text'], axis=1).columns
    df[cols_with_scores] = (df[cols_with_scores] > threshold).astype(int)
    print('Shape of unintended data:', df.shape)
    return df


def read_ruddit(folder_ruddit: str) -> pd.DataFrame:
    folder_ruddit = Path(folder_ruddit)
    df = pd.read_csv(folder_ruddit)
    df = df.rename({'txt': 'comment_text', 'comment_id': 'id'}, axis=1)

    columns = ['id', 'comment_text', 'offensiveness_score']
    df = df.loc[:, columns]

    df = df[df['comment_text'] != '[deleted]']
    df.loc[df['offensiveness_score'] < 0] = 0
    print('Shape of ruddit data:', df.shape)
    return df


def make_sample(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([
        df.head(20),
        df.sample(n=40),
        df.tail(20)
        ])


def calculate_score(df: pd.DataFrame) -> pd.DataFrame:
    columns = ['toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'severe_toxic']
    df['severe_toxic'] *= 2
    df['offensiveness_score'] = 0

    for col in columns:
        df['offensiveness_score'] += df[col]

    df['offensiveness_score'] = df['offensiveness_score'] / len(columns)
    df = df[['id', 'comment_text', 'offensiveness_score']]
    return df


def process_text(full_line: str, full_process: bool = False) -> str:
    full_line = str(full_line)
    full_line = re.sub(r'https.*[^ ]', 'URL', full_line)
    full_line = re.sub(r'http.*[^ ]', 'URL', full_line)
    full_line = re.sub(r'@([^ ]*)', '@USER', full_line)
    if full_process:
        full_line = re.sub(r'#([^ ]*)', r'\1', full_line)
        full_line = emoji.demojize(full_line)
        full_line = re.sub(r'(:.*?:)', r' \1 ', full_line)
        full_line = re.sub(' +', ' ', full_line)
    return full_line


@click.command()
@click.option('--folder_toxic', help='Path to folder of jigsaw toxic comment classification', default='jigsaw-toxic-comment-classification-challenge')
@click.option('--folder_unintended', help='Path to folder of jigsaw unintended bias', default='jigsaw-unintended-bias-in-toxicity-classification')
@click.option('--folder_ruddit', help='Path to ruddit dataset (full path to file)', default='ruddit/Dataset/ruddit_with_text.csv')
@click.option('--output', help='Output path', default='jigsaw_train.csv')
@click.option('--unintended_threshold', help='Threshold for unintended dataset classification', default=0.5)
@click.option('--text_process/--no-text_process', help='Full text preprocess', default=False)
@click.option('--preprocess_type', type=click.Choice(['sparse', 'dense'], case_sensitive=True),
              help='"dense" or "sparse", see documentation for details', default='dense')
def main(
        folder_toxic: str,
        folder_unintended: str,
        output: str,
        unintended_threshold: float,
        folder_ruddit: str,
        text_process: bool,
        preprocess_type: str
        ) -> None:
    """Tool to convert test and train dataset from
    https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data?select=train.csv.zip
    and
    https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data?select=all_data.csv
    to train dataset.
    """
    print('Toxic Comment Classification Challenge')
    toxic_df = read_toxic_data(folder_toxic)

    print('Ruddit dataset')
    ruddit = read_ruddit(folder_ruddit)
    
    print('Jigsaw Unintended Bias in Toxicity Classification')
    if preprocess_type == 'dense':
        toxic_df = calculate_score(toxic_df)
        unintented_df = read_unintended_data_dense(folder_unintended)
        total = pd.concat([toxic_df, unintented_df, ruddit])
    else:
        unintented_df = read_unintended_data_sparse(folder_unintended, unintended_threshold)
        total = pd.concat([toxic_df, unintented_df])
        total = calculate_score(total)
        total = pd.concat([total, ruddit])

    # total = unintented_df
    total.loc[total['offensiveness_score'] > 1, 'offensiveness_score'] = 1

    print('Data preprocessing')
    tqdm.pandas()
    total['comment_text'] = total['comment_text'].progress_apply(process_text, full_process=text_process)

    num_duplicates = total.duplicated(subset='comment_text').sum()
    if num_duplicates > 0:
        print(f'Founded {num_duplicates} duplicated rows, will be deleted')
        total = total.drop_duplicates(subset='comment_text')

    print('Shape of united data after filtering:', total.shape)
    total.to_csv(output, index=False)

    print('Making sample dataset to check quality')
    sample = make_sample(total)
    sample.to_csv('sample.csv', index=False)


if __name__ == '__main__':
    main()
