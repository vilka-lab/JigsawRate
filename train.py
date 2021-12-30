from model import JigsawModel
from pathlib import Path
from dataset import get_loader
import click
import torch
import pandas as pd
from transformers import AutoTokenizer


@click.command()
@click.option('--data', help='Path to train data', default='data/jigsaw_train.csv')
@click.option('--lr', help='Learning rate', default=1e-3)
@click.option('--weight_decay', default = 5e-3)
@click.option('--epochs', help='Number of epochs', default=30)
@click.option('--resume/--no-resume', help='Resume training process', default=False)
@click.option('--num_workers', help='Number of workers', default=2)
@click.option('--batch_size', default=8)
def main(
        data: str,
        lr: float,
        weight_decay: float,
        epochs: int,
        resume: bool,
        num_workers: int,
        batch_size: int
        ):
    # torch.backends.cudnn.benchmark = True
    # torch.autograd.set_detect_anomaly(False)
    # torch.autograd.profiler.profile(False)
    # torch.autograd.profiler.emit_nvtx(False)
    model_weights = 'experiment/last.pth'
    model = JigsawModel()
    model_path = Path(model_weights)
    if resume and model_path.exists():
        model.load_model(model_path, load_train_info=resume)
        print('Model loaded from', model_path)

    tokenizer = AutoTokenizer.from_pretrained('GroNLP/hateBERT',
                                              model_max_length=256)

    path = Path('data').joinpath('validation_data.csv')
    df = pd.read_csv(path)
    val_loaders = [
        get_loader(df['less_toxic'], tokenizer, num_workers=num_workers,
                   batch_size=batch_size),
        get_loader(df['more_toxic'], tokenizer, num_workers=num_workers,
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
        weight_decay=weight_decay
        )


if __name__ == "__main__":
    main()
