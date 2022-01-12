import os
import datetime
import pandas as pd
from abc import ABCMeta, abstractmethod
from typing import Optional
from tqdm import tqdm

import torch
from torch import nn
from torch.cuda import is_available
from torch.utils.data import DataLoader
from torch.optim import Optimizer

__all__ = ['Experiment', 'save_params', 'to_csv']


class Experiment(metaclass=ABCMeta):
    '''
    実験の抽象クラス.

    Attribute
    ---------
    max_epoch: int
        学習させるエポックの最大値.
    batch_size: int
        バッチサイズ.
    model: torich.nn.Module
        学習させるモデル.
    optimizer: torch.optim.Optimizer
        学習アルゴリズム.
    data_dir: str = './data'
        データを保存するディレクトリ.
    csv_dir: str = './results/csv'
        学習過程のcsvを保存するディレクトリ.
    csv_name: Optional[str] = None,
        学習過程のcsvファイル名.
    prm_dir: str ='./results/prm'
        学習したモデルのパラメータを保存するディレクトリ.
    prm_name: Optional[str] = None,
        学習したモデルのパラメータ名.
    '''
    def __init__(
        self,
        model     : nn.Module,
        optimizer : Optimizer,
        max_epoch : int,
        batch_size: int,
        data_dir  : str = './data',
        csv_dir   : str = './results/csv',
        csv_name  : Optional[str] = None,
        prm_dir   : str = './results/prm',
        prm_name  : Optional[str] = None,
    ) -> None:

        self.model     : nn.Module     = model
        self.optimizer : Optimizer     = optimizer
        self.max_epoch : int           = max_epoch
        self.batch_size: int           = batch_size
        self.data_dir  : str           = data_dir
        self.csv_dir   : str           = csv_dir
        self.csv_name  : Optional[str] = csv_name
        self.prm_dir   : str           = prm_dir
        self.prm_name  : Optional[str] = prm_name

        # GPUを使用するなら'cuda:0', そうでないなら'cpu'を値として持つ.
        self.device: str = 'cuda:0' if is_available() else 'cpu'

    def __call__(self) -> nn.Module:
        return self.run()

    @abstractmethod
    def prepare_data(self) -> tuple[DataLoader, DataLoader]:
        ...

    @abstractmethod
    def _train(self, train_loader: DataLoader) -> list[float]:
        ...
    
    @abstractmethod
    def _eval(self, test_loader: DataLoader) -> list[float]:
        ...

    def run(self) -> nn.Module:
        '''
        実験を実行する.
        '''

        # モデルが存在するならロードする.
        if os.path.isfile(os.path.join(self.prm_dir, self.prm_name)):
            print(f'{os.path.join(self.prm_dir, self.prm_name)} already exists.')
            self.model.load_state_dict(torch.load(os.path.join(self.prm_dir, self.prm_name)))
        self.model = self.model.to(self.device)

        # DataLoaderを準備する.
        train_loader, test_loader = self.prepare_data()

        # 各Epochで必要な情報を保持しておくリスト.
        record: list[list[float]] = []

        # 学習とテストのforループ.
        for epoch in tqdm(range(self.max_epoch)):
            # ネットワークの訓練
            self.model.train()
            _t_list: list[float] = self._train(train_loader)
            
            # ネットワークのテスト
            self.model.eval()
            _e_list: list[float] = self._eval(test_loader)

            _row = [v for v in _t_list + _e_list]
            print()
            print(f'epoch {epoch}:', _row)
            record.append(_row)
        
        save_params(self.model, prm_dir=self.prm_dir, prm_name=self.prm_name)
        to_csv(record, csv_dir=self.csv_dir, csv_name=self.csv_name)
        return self.model


def save_params(
    model: nn.Module,
    prm_dir: str,
    prm_name: Optional[str]=None
) -> None:

    if not os.path.isdir(prm_dir):
        os.makedirs(prm_dir)
    if prm_name is None:
        prm_name = f'{_now2str()}.prm'
    torch.save(model.to('cpu').state_dict(), os.path.join(prm_dir, prm_name))


def to_csv(
    record: list[list[float]],
    csv_dir: str,
    csv_name: Optional[str]=None
) -> None:

    df: pd.DataFrame = pd.DataFrame(record, columns=None, index=None)
    if not os.path.isdir(csv_dir):
        os.makedirs(csv_dir)
    if csv_name is None:
        csv_name = f'{_now2str()}.csv'

    print(f'{os.path.join(csv_dir, csv_name)} already exists.')
    df.to_csv(os.path.join(csv_dir, csv_name), mode='a', header=None, index=None)


def _now2str() -> str:
    tzinfo = datetime.timezone(datetime.timedelta(hours=9))
    dt_now = datetime.datetime.now(tz=tzinfo)
    _year  : str = str(dt_now.year).zfill(2)
    _month : str = str(dt_now.month).zfill(2)
    _day   : str = str(dt_now.day).zfill(2)
    _hour  : str = str(dt_now.hour).zfill(2)
    _minute: str = str(dt_now.minute).zfill(2)
    _second: str = str(dt_now.second).zfill(2)
    _now: str = f'{_year}-{_month}-{_day}-{_hour}:{_minute}:{_second}'
    return _now
