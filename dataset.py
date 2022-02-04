from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
ABS_PATH = Path(__file__).parent.absolute()
from typing import Union, Tuple, Optional
import transformers
from transformers import AutoTokenizer
import numpy as np
Tokenizer = transformers.models.bert.tokenization_bert_fast.BertTokenizerFast


class JigsawDataset(Dataset):
    def __init__(self, data: Union[pd.DataFrame, pd.Series],
                 tokenizer: Tokenizer) -> None:
        """data - pass pd.DataFrame for train dataset, else pd.Series"""
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.train = isinstance(data, pd.DataFrame)
        # if self.train:
        #     self._downsampling()
        self.data_len = self.data.shape[0]


    def _downsampling(self):
        toxic_samples = self.data[self.data['offensiveness_score'] > 0]
        nontoxic_samples = self.data[self.data['offensiveness_score'] == 0]
        self.data = pd.concat([
            toxic_samples,
            nontoxic_samples.sample(n=toxic_samples.shape[0] // 2)
        ])
        self.data = self.data.sample(frac=1)


    def _sample(self, row: pd.Series) -> Optional[pd.Series]:
        if not self.train:
            return None

        ix = np.random.randint(0, self.data_len)
        new_row = self.data.iloc[ix]
        count = 0
        while row['offensiveness_score'] == new_row['offensiveness_score']:
            ix = np.random.randint(0, self.data_len)
            new_row = self.data.iloc[ix]
            count += 1
            if count == 1000:
                raise ValueError('Something wrong, sampling is broken')
        return new_row


    def __len__(self) -> int:
        return len(self.data)


    def _tokenize(self, txt: str) -> dict:
        vector = self.tokenizer(str(txt), padding='max_length', truncation=True,
                                return_tensors="pt")
        vector = {key: val.flatten() for key, val in vector.items()}
        return vector


    def __getitem__(self, index: int) -> Tuple[dict, torch.Tensor]:
        row = self.data.iloc[index]
        second_row = self._sample(row)

        if self.train:
            if row['offensiveness_score'] > second_row['offensiveness_score']:
                row, second_row = second_row, row

            less_toxic_vector = self._tokenize(row['comment_text'])
            more_toxic_vector = self._tokenize(second_row['comment_text'])
            less_toxic_target = torch.tensor(row['offensiveness_score'], dtype=torch.float32)
            more_toxic_target = torch.tensor(second_row['offensiveness_score'], dtype=torch.float32)
        else:
            less_toxic_vector = self._tokenize(row)
            more_toxic_vector = self._tokenize('')
            less_toxic_target = torch.tensor(0, dtype=torch.float32)
            more_toxic_target = torch.tensor(0, dtype=torch.float32)
        return less_toxic_vector, less_toxic_target, more_toxic_vector, more_toxic_target


def get_loader(
        df: Union[pd.DataFrame, pd.Series],
        tokenizer: Tokenizer,
        batch_size: int = 32,
        num_workers: int = 2,
        train: bool = False
        ) -> torch.utils.data.DataLoader:

    dataset = JigsawDataset(df, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        drop_last=train
        )
    return dataloader


def test_dataset() -> None:
    tokenizer = AutoTokenizer.from_pretrained('GroNLP/hateBERT', model_max_length=256)

    print('Testing validation data')
    path = Path('data').joinpath('validation_data.csv')
    df = pd.read_csv(path)

    loader_1 = get_loader(df['less_toxic'], tokenizer, num_workers=0)
    for vector, _, _, _ in tqdm(loader_1):
        pass
    print(vector['input_ids'].shape, vector['attention_mask'].shape)

    loader_2 = get_loader(df['more_toxic'], tokenizer, num_workers=0)
    for vector, _, _, _ in tqdm(loader_2):
        pass
    print(vector['input_ids'].shape, vector['attention_mask'].shape)

    print('Testing train data')
    path = Path('data').joinpath('sample.csv')
    df = pd.read_csv(path)
    train_loader = get_loader(df, tokenizer, num_workers=0, train=True)
    for less_toxic_vector, less_toxic_target, more_toxic_vector, more_toxic_target in tqdm(train_loader):
        pass
    print(less_toxic_vector['input_ids'].shape,
          less_toxic_vector['attention_mask'].shape,
          less_toxic_target.shape)
    print(less_toxic_target)
    print(more_toxic_target)


if __name__ == '__main__':
    test_dataset()
