from typing import Optional
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader

import subprocess
import os
from time import time
from tqdm import tqdm
import glob
import pathlib
import re

from experiments import Experiment

__all__ = ['ExperimentIMDb']


def _text2ids(
    text: str,
    vocab_dict: dict[str, int]
) -> list[int]:

    remove_marks_regex = re.compile("[,\.\(\)\[\]\*:;]|<.*?>")
    shift_marks_regex = re.compile("([?!])")

    # !?以外の記号の削除
    text = remove_marks_regex.sub("", text)
    # !?と単語の間にスペースを挿入
    text = shift_marks_regex.sub(r" \1 ", text)
    tokens = text.split()
    return [vocab_dict.get(token, 0) for token in tokens]


def _list2tensor(
    token_idxes: list[int],
    max_len: int=100,
    padding: bool=True
) -> tuple[torch.Tensor, int]:

    if len(token_idxes) > max_len:
        token_idxes = token_idxes[:max_len]
    n_tokens = len(token_idxes)
    if padding:
        token_idxes = token_idxes + [0] * (max_len - len(token_idxes))
    return torch.tensor(token_idxes, dtype=torch.int64), n_tokens


class IMDbDataset(Dataset):
    def __init__(
        self,
        dir_path: str,
        train   : bool=True,
        max_len : int=100,
        padding : bool=True
    ) -> None:

        self.max_len: int = max_len
        self.padding: bool = padding

        path = pathlib.Path(dir_path)
        vocab_path = path.joinpath("imdb.vocab")

        # ボキャブラリファイルを読み込み、行ごとに分割
        self.vocab_array = vocab_path.open() \
            .read().strip().splitlines()
        # 単語をキーとし、値がIDのdictを作る
        self.vocab_dict = dict((w, i + 1) for (i, w) in enumerate(self.vocab_array))

        if train:
            target_path = path.joinpath("train")
        else:
            target_path = path.joinpath("test")
        pos_files = sorted(glob.glob(
            str(target_path.joinpath("pos/*.txt"))))
        neg_files = sorted(glob.glob(
            str(target_path.joinpath("neg/*.txt"))))
        # posは1, negは0のlabelを付けて
        # (file_path, label)のtupleのリストを作成
        self.labeled_files = \
            list(zip([0] * len(neg_files), neg_files)) + \
            list(zip([1] * len(pos_files), pos_files))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab_array)

    def __len__(self) -> int:
        return len(self.labeled_files)

    def __getitem__(self, index: int) -> tuple[str, int, int]:
        label, f = self.labeled_files[index]
        # ファイルのテキストデータを読み取って小文字に変換
        data = open(f).read().lower()
        # テキストデータをIDのリストに変換
        data = _text2ids(data, self.vocab_dict)
        # IDのリストをTensorに変換
        data, n_tokens = _list2tensor(data, self.max_len, self.padding)
        return data, label, n_tokens


class ExperimentIMDb(Experiment):
    def __init__(
        self,
        model     : nn.Module,
        optimizer : Optimizer,
        max_epoch : int,
        batch_size: int,
        data_dir  : str = './data',
        csv_dir   : str = './results/csv/imdb',
        csv_name  : Optional[str] = None,
        prm_dir   : str = './results/prm/imdb',
        prm_name  : Optional[str] = None,
        download  : bool=True
    ) -> None:

        super().__init__(
            model,
            optimizer,
            max_epoch=max_epoch,
            batch_size=batch_size,
            data_dir=data_dir,
            csv_dir=csv_dir,
            csv_name=csv_name,
            prm_dir=prm_dir,
            prm_name=prm_name
        )
        self.download = download

    def prepare_data(self) -> tuple[DataLoader, DataLoader]:
        if self.download:
            if not os.path.exists(os.path.join(self.data_dir, 'aclImdb')):
                subprocess.run(args=[
                    'wget',
                    '-P',
                    self.data_dir,
                    'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
                ])

                subprocess.run(args=[
                    'tar',
                    'xf',
                    os.path.join(self.data_dir, 'aclImdb_v1.tar.gz'),
                    '-C',
                    self.data_dir
                ])
            else:
                print('Files already downloaded and verified')

        train_data:   Dataset    = IMDbDataset(os.path.join(self.data_dir, 'aclImdb'), train=True)
        test_data:    Dataset    = IMDbDataset(os.path.join(self.data_dir, 'aclImdb'), train=False)
        train_loader: DataLoader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader:  DataLoader = DataLoader(test_data, batch_size=32, shuffle=True)
        return train_loader, test_loader

    def _train(self, train_loader: DataLoader) -> list[float]:
        model = self.model
        optimizer = self.optimizer

        criterion = nn.BCEWithLogitsLoss()
        i           : int   = 0       # ステップ数
        total       : int   = 0       # 全ての訓練データの数
        correct     : int   = 0       # 正しく分類された訓練データ数
        running_loss: float = 0.0     # 経験損失の合計
        tic         : float = time()  # 訓練開始の時間 (UNIX時間)

        for inputs, labels, l in tqdm(train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            l = l.to(self.device)

            outputs: torch.Tensor = model(inputs, l=l)
            loss = criterion(outputs, labels.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total += labels.size(0)
            predicted: torch.Tensor = (outputs > 0).long()
            correct += (labels == predicted).sum().item()
            i += 1
            
        train_loss: float = running_loss / i
        train_acc : float = correct / total
        train_time: float = time() - tic
        return train_loss, train_acc, train_time
    
    def _eval(self, test_loader: DataLoader) -> list[float]:
        model = self.model

        criterion = nn.BCEWithLogitsLoss()
        i           : int   = 0    # ステップ数
        total       : int   = 0    # 全てのテストデータの数
        correct     : int   = 0    # 正しく分類されたテストデータの数
        running_loss: float = 0.0  # 予測損失の合計

        with torch.no_grad():
            for inputs, labels, l in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                l = l.to(self.device)

                outputs = model(inputs, l=l)
                loss: torch.Tensor = criterion(outputs, labels.float())

                running_loss += loss.item()
                predicted: torch.Tensor = (outputs > 0).long()
                total += labels.size(0)
                correct += (labels == predicted).sum().item()
                i += 1
        
        test_loss: float = running_loss / i
        test_acc : float = correct / total
        return test_loss, test_acc
