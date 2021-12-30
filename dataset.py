from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
ABS_PATH = Path(__file__).parent.absolute()
from typing import Union, Tuple
import transformers
from transformers import AutoTokenizer
Tokenizer = transformers.models.bert.tokenization_bert_fast.BertTokenizerFast


class JigsawDataset(Dataset):
    def __init__(self, data: Union[pd.DataFrame, pd.Series],
                 tokenizer: Tokenizer) -> None:
        """data - pass pd.DataFrame for train dataset, else pd.Series"""
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.train = isinstance(data, pd.DataFrame)


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, index: int) -> Tuple[dict, torch.Tensor]:
        row = self.data.iloc[index]

        if self.train:
            txt = row['comment_text']
            target = row['offensiveness_score']
        else:
            txt = row
            target = 0

        vector = self.tokenizer(txt, padding='max_length', truncation=True,
                                return_tensors="pt")
        vector = {key: val.flatten() for key, val in vector.items()}
        target = torch.tensor(target, dtype=torch.float32)
        return vector, target


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
    for vector, _ in tqdm(loader_1):
        pass
    print(vector['input_ids'].shape, vector['attention_mask'].shape)

    loader_2 = get_loader(df['more_toxic'], tokenizer, num_workers=0)
    for vector, _ in tqdm(loader_2):
        pass
    print(vector['input_ids'].shape, vector['attention_mask'].shape)

    print('Testing train data')
    path = Path('data').joinpath('sample.csv')
    df = pd.read_csv(path)
    train_loader = get_loader(df, tokenizer, num_workers=0, train=True)
    for vector, target in tqdm(train_loader):
        pass
    print(vector['input_ids'].shape, vector['attention_mask'].shape, target.shape)
    print(target)


if __name__ == '__main__':
    test_dataset()
