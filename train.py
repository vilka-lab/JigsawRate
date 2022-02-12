from model import JigsawModel
from pathlib import Path
from dataset import get_loader
import click
import torch
import pandas as pd
from transformers import AutoTokenizer
import numpy as np
from data.convert_jigsaw import process_text
from tqdm import tqdm


def set_seed(seed=42):
    """Set all seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@click.command()
@click.option('--data', help='Path to train data', default='data/jigsaw_train.csv')
@click.option('--lr', help='Learning rate', default=1e-3)
@click.option('--weight_decay', default = 1e-3)
@click.option('--epochs', help='Number of epochs', default=30)
@click.option('--resume/--no-resume', help='Resume training process', default=False)
@click.option('--num_workers', help='Number of workers', default=2)
@click.option('--batch_size', default=8)
@click.option('--freeze/--no-freeze', help='Freeze Bert layers', default=False)
@click.option('--random_state', default=42)
@click.option('--force_lr/--no-force_lr', help='Force set learning rate', default=False)
@click.option('--max_length', help='Max_length param in tokenizer', default=128)
@click.option('--optimizer', type=click.Choice(['Adam', 'AdamW', 'SGD'], case_sensitive=True),
              help='Adam/AdamW, SGD', default='Adam')
@click.option('--model_name', default='GroNLP/hateBERT')
@click.option('--text_process/--no-text_process', help='Full text preprocess', default=False)
@click.option('--objective', type=click.Choice(['margin', 'bce'], case_sensitive=True),
              help='Margin loss (margin) or BCELoss(bce)', default='margin')
def main(
        data: str,
        lr: float,
        weight_decay: float,
        epochs: int,
        resume: bool,
        num_workers: int,
        batch_size: int,
        freeze: bool,
        random_state: int,
        force_lr: bool,
        max_length: int,
        optimizer: str,
        model_name: str,
        text_process: bool,
        objective: str
        ):
    set_seed(random_state)
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    model_weights = 'experiment/last.pth'
    model = JigsawModel(model_name=model_name)
    model_path = Path(model_weights)
    if resume and model_path.exists():
        model.load_model(model_path, load_train_info=resume)
        print('Model loaded from', model_path)

    if freeze:
        model.freeze()
    else:
        model.unfreeze()

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              model_max_length=max_length)

    path = Path('data').joinpath('validation_data.csv')
    df = pd.read_csv(path)
    print('Creating validation datasets')
    tqdm.pandas()
    val_loaders = [
        get_loader(df['less_toxic'].progress_apply(process_text, full_process=text_process), tokenizer, num_workers=num_workers,
                   batch_size=batch_size),
        get_loader(df['more_toxic'].progress_apply(process_text, full_process=text_process), tokenizer, num_workers=num_workers,
                   batch_size=batch_size)
        ]

    path = Path(data)
    df = pd.read_csv(path)
    train_loader = get_loader(df, tokenizer,  num_workers=num_workers,
                              batch_size=batch_size, train=True)

    model.fit(
        num_epochs=epochs,
        train_loader=train_loader,
        val_loaders=val_loaders,
        folder='experiment',
        learning_rate=lr,
        force_lr=force_lr,
        weight_decay=weight_decay,
        optimizer=optimizer,
        objective=objective
        )


if __name__ == "__main__":
    main()
