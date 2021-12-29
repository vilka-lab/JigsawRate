import pandas as pd
import click
from pathlib import Path
from typing import List


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


def read_unintended_data(folder_unintended: str, threshold: float) -> pd.DataFrame:
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
    df[cols_with_scores] = (df[cols_with_scores] > 0.5).astype(int)
    print('Shape of unintended data:', df.shape)
    return df


def read_ruddit(folder_ruddit: str) -> pd.DataFrame:
    folder_ruddit = Path(folder_ruddit)
    df = pd.read_csv(folder_ruddit)
    df = df.rename({'txt': 'comment_text'}, axis=1)

    columns = ['comment_id', 'comment_text', 'offensiveness_score']
    df = df.loc[:, columns]

    df = df[df['comment_text'] != '[deleted]']
    df.loc[df['offensiveness_score'] < 0] = 0
    return df


def make_sample(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([
        df.head(20),
        df.sample(n=40),
        df.tail(20)
        ])


def calculate_score(df: pd.DataFrame) -> pd.DataFrame:
    columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df['offensiveness_score'] = 0
    for col in columns:
        df['offensiveness_score'] += df[col] if col != 'severe_toxic' else 1.5 * df[col]

    df['offensiveness_score'] = df['offensiveness_score'] / df['offensiveness_score'].max()
    df.drop(columns, axis=1, inplace=True)
    return df


@click.command()
@click.option('--folder_toxic', help='Path to folder of jigsaw toxic comment classification', default='jigsaw-toxic-comment-classification-challenge')
@click.option('--folder_unintended', help='Path to folder of jigsaw unintended bias', default='jigsaw-unintended-bias-in-toxicity-classification')
@click.option('--folder_ruddit', help='Path to ruddit dataset (full path to file)', default='ruddit/Dataset/ruddit_with_text.csv')
@click.option('--output', help='Output path', default='jigsaw_train.csv')
@click.option('--unintended_threshold', help='Threshold for unintended dataset classification', default=0.5)
def main(
        folder_toxic: str,
        folder_unintended: str,
        output: str,
        unintended_threshold: float,
        folder_ruddit: str
        ) -> None:
    """Tool to convert test and train dataset from
    https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data?select=train.csv.zip
    and
    https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data?select=all_data.csv
    to train dataset.
    """
    # --------------------------------------------------------
    print('Toxic Comment Classification Challenge')
    toxic_df = read_toxic_data(folder_toxic)

    # --------------------------------------------------------
    print('Jigsaw Unintended Bias in Toxicity Classification')
    unintented_df = read_unintended_data(folder_unintended, unintended_threshold)

    # --------------------------------------------------------
    total = pd.concat([toxic_df, unintented_df])
    total = calculate_score(total)

    print('Ruddit dataset')
    ruddit = read_ruddit(folder_ruddit)
    total = pd.concat([total, ruddit])

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