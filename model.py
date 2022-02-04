# -*- coding: utf-8 -*-

import torch
import pickle
from datetime import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from transformers import AutoModel
from typing import Dict, Union, Any, Callable, Optional, List
Vector = Dict[str, torch.Tensor]
LoaderList = List[torch.utils.data.DataLoader]


class JigsawModel(torch.nn.Module):
    def __init__(self, model_name: str = 'GroNLP/hateBERT',
                 device: Optional[str] = None,
                 pretrained: Optional[str] = None) -> None:
        super().__init__()
        if pretrained is None:
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            self.backbone = AutoModel.from_pretrained(pretrained)

        self.fc = torch.nn.Linear(768, 1)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.to(self.device)
        self.epoch = 0
        self.resume = False


    def forward(self, vector: Vector) -> torch.Tensor:
        out = self.backbone(
            input_ids=vector['input_ids'],
            attention_mask=vector['attention_mask'],
            output_hidden_states=False
            )
        outputs = self.fc(out[1])
        return outputs


    def freeze(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False


    def unfreeze(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


    def save_model(self, path: Union[Path, str]) -> None:
        path = str(path)
        checkpoint = {
            'epoch': self.epoch,
            'model': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_sched': self.scheduler}
        torch.save(checkpoint, path)


    def load_model(self, path: Union[Path, str],
                   load_train_info: bool = False) -> None:
        path = str(path)

        if self.device == 'cpu':
            checkpoint = torch.load(path, map_location=torch.device(self.device))
        else:
            checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model'])

        if load_train_info:
            self.create_optimizer()
            self.epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler = checkpoint['lr_sched']
            self.resume = True


    def load_file(self, filename: Union[Path, str]) -> Any:
        filename = str(filename)
        try:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
        except FileNotFoundError:
            self.save_file(filename, [])
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
        return obj


    def save_file(self, filename: Union[Path, str], obj: Any) -> None:
        filename = str(filename)
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)


    def load_storage(self, path: Path) -> dict:
        obj = {}
        obj['train_losses'] = self.load_file(path.joinpath('train_losses'))
        obj['test_metrics'] = self.load_file(path.joinpath('test_metrics'))
        return obj


    def save_storage(self, path: Path, obj: dict) -> None:
        self.save_file(path.joinpath('train_losses'), obj['train_losses'])
        self.save_file(path.joinpath('test_metrics'), obj['test_metrics'])


    def save_each_period(self, path: Union[Path, str], period: int = 60*60) -> Callable:
        now = datetime.now()
        path = str(path)

        def save_model() -> None:
            nonlocal now
            delta = datetime.now() - now
            if delta.total_seconds() > period:
                self.save_model(path)
                now = datetime.now()
        return save_model


    @staticmethod
    def metric(
            less_toxic: Union[np.array, torch.Tensor],
            more_toxic: Union[np.array, torch.Tensor]
            ) -> float:
        return float((less_toxic < more_toxic).mean())


    def create_optimizer(self, learning_rate: float = 1e-3,
                         weight_decay: float = 1e-3,
                         optimizer: str = 'Adam') -> None:
        main_params = {
            'lr': learning_rate,
            'weight_decay': weight_decay}
        sgd_params = main_params.copy()
        sgd_params['momentum'] = 0.9

        optimizers = {
            'Adam': (torch.optim.Adam, main_params),
            'AdamW': (torch.optim.AdamW, main_params),
            'SGD': (torch.optim.SGD, sgd_params)
            }

        self.optimizer = optimizers[optimizer][0](
            self.parameters(),
            **optimizers[optimizer][1]
            )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=2)


    def fit(
            self,
            num_epochs: int,
            train_loader: torch.utils.data.DataLoader,
            val_loaders: Optional[LoaderList],
            folder: str = 'experiment',
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-3,
            save_period: int = 60*60,
            force_lr: bool = False,
            optimizer: str = 'Adam'
            ) -> dict:
        if not Path(folder).exists():
            os.mkdir(folder)
        path = Path('./').joinpath(folder)
        storage = self.load_storage(path)

        model_path = path.joinpath('last.pth')

        self.saver = self.save_each_period(model_path, save_period)
        # self.loss = torch.nn.BCEWithLogitsLoss()
        self.loss = torch.nn.MarginRankingLoss(margin=1)
        if not self.resume or force_lr:
            self.create_optimizer(learning_rate, weight_decay, optimizer)
        self.scaler = torch.cuda.amp.GradScaler()

        # in respect for --resume flag
        for i in range(num_epochs - self.epoch):
            if val_loaders is not None:
                self._test_loop(val_loaders, storage=storage)

            self._train_loop(train_loader, storage=storage)
            self.epoch += 1

            self.save_model(model_path)
            self.save_storage(path, storage)
            self.visualize(folder, storage)

            print('-'*30)
        return storage


    def _train_loop(
            self,
            train_loader: torch.utils.data.DataLoader,
            storage: dict
            ) -> None:
        self.train()
        losses = []
        with tqdm(total=len(train_loader)) as progress_bar:
            for less_toxic_vector, _, more_toxic_vector, _ in train_loader:
                self.optimizer.zero_grad()

                less_toxic_vector = {key: val.to(self.device) for key, val in less_toxic_vector.items()}
                more_toxic_vector = {key: val.to(self.device) for key, val in more_toxic_vector.items()}

                with torch.cuda.amp.autocast():
                    less_toxic_preds = self.forward(less_toxic_vector)
                    more_toxic_preds = self.forward(more_toxic_vector)
                    loss_val = self.loss(
                        less_toxic_preds.flatten(),
                        more_toxic_preds.flatten(),
                        -1 * torch.ones(less_toxic_preds.shape[0], dtype=torch.float32).to(self.device)
                        )

                self.scaler.scale(loss_val).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                losses.append(loss_val.item())
                self.saver()

                loss_val = np.mean(losses)

                progress_bar.update()
                progress_bar.set_description('Epoch {}: loss = {:.4f}'.format(
                                             self.epoch, loss_val))
        storage['train_losses'].append(loss_val)
        self.scheduler.step(loss_val)


    def _test_loop(self, val_loaders: LoaderList, storage: dict) -> None:
        self.eval()
        less_toxic = self.predict(val_loaders[0])
        more_toxic = self.predict(val_loaders[1])
        print('validation score examples')
        print(less_toxic[:10])
        print(more_toxic[:10])
        metric_val = self.metric(less_toxic, more_toxic)
        print('Validation metric:', metric_val)
        storage['test_metrics'].append(metric_val)


    def visualize(self, folder: Union[Path, str], storage: Optional[dict]) -> None:
        path = Path('./').joinpath(folder)

        if storage is None:
            storage = self.load_storage(path)

        plt.style.use('fivethirtyeight')

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(storage['train_losses'], c='y', label='train loss')
        axes[0].set_title('losses')
        axes[1].plot(storage['test_metrics'], c='b', label='validation metric')
        axes[1].set_title('metrics')
        plt.legend()
        fig.savefig(path.joinpath('results.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)


    def predict(self, dataloader: torch.utils.data.DataLoader,
                verbose: bool = True) -> np.array:
        self.eval()
        result = []
        with torch.no_grad():
            for vector, _, _, _ in tqdm(dataloader, disable=(not verbose)):
                vector = {key: val.to(self.device) for key, val in vector.items()}
                preds = self.forward(vector)
                result.append(preds.cpu().detach().numpy())
        result = np.concatenate(result)
        return result


class MarginLoss():
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def __call__(
            self,
            inp: Union[torch.Tensor, torch.Tensor.cuda],
            target: Union[torch.Tensor, torch.Tensor.cuda]
            ) -> torch.Tensor:
        if inp.shape[0] != target.shape[0]:
            raise ValueError(f'Wrong shape of input data, {inp.shape}, {target.shape}')

        weights = ((inp.reshape(-1, 1) > inp.reshape(1, -1)) & (target.reshape(-1, 1) < target.reshape(1, -1))).double()
        loss = (inp.reshape(-1, 1) - inp.reshape(1, -1) + self.delta) * weights

        return loss.mean()


if __name__ == '__main__':
    loss = MarginLoss()
    a = torch.tensor([[0.3, 0.2, 0.6, 1.2]])
    b = torch.tensor([[0.2, 0.1, 0.3, 1.7]])
    print(a, b, loss(a, b))

    a = torch.tensor([[0.7, 1.2, 0.6, 1.2]])
    b = torch.tensor([[0.2, 0.1, 0.3, 1.7]])
    print(a, b, loss(a, b))