import os
import string
import subprocess
from tqdm import tqdm
from time import time
from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

from experiments import Experiment

__all__ = [
    'ExperimentShakespeare',
]


def _str2ints(s: str) -> list[int]:
    '''
    文字列をint型のリストに変換する.
    '''

    # 全てのascii文字で辞書を作る.
    all_chars: str = string.printable
    vocab_dict: dict[str, int] = dict((c, i) for (i, c) in enumerate(all_chars))
    
    return [vocab_dict[c] for c in s]


def _ints2str(x: list[int]) -> str:
    '''
    int型のリストを文字列に変換する.
    '''
    return ''.join([string.printable[i] for i in x])


class ShakespeareDataset(Dataset):
    def __init__(self, path: str, chunk_size: int=200) -> None:
        #ファイルを読み込みint型のリストに変換する.
        data = _str2ints(open(path).read().strip())
        data = torch.tensor(data, dtype=torch.int64).split(chunk_size)
        if len(data[-1]) < chunk_size:
            data = data[:-1]
        
        self.data: torch.Tensor = data
        self.n_chunks = len(self.data)
    
    def __len__(self):
        return self.n_chunks
    
    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]


class ExperimentShakespeare(Experiment):
    def __init__(
        self,
        model     : nn.Module,
        optimizer : Optimizer,
        max_epoch : int,
        batch_size: int,
        data_dir  : str = './data',
        data_name : str = 'shakespeare/tinyshakespeare.txt',
        csv_dir   : str = './results/csv/shakespeare',
        csv_name  : Optional[str] = None,
        prm_dir   : str = './results/prm/shakespeare',
        prm_name  : Optional[str] = None,
        token     : Optional[str] = None,
        download  : bool = True
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
            prm_name=prm_name,
            token=token
        )
        self._vocab_size: int  = len(string.printable)
        self.data_name  : str  = data_name
        self.download   : bool = download

    def prepare_data(self) -> tuple[DataLoader, DataLoader]:
        if self.download:
            if not os.path.exists(os.path.join(self.data_dir, 'shakespeare')):
                os.makedirs(os.path.join(self.data_dir, 'shakespeare'))
                subprocess.run(args=[
                    'wget',
                    '-P',
                    os.path.join(self.data_dir, 'shakespeare'),
                    'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
                ])
                subprocess.run(args=[
                    'mv',
                    os.path.join(self.data_dir, 'shakespeare/input.txt'),
                    os.path.join(self.data_dir, 'shakespeare/tinyshakespeare.txt')
                ])
            else:
                print('Files already downloaded and verified')

        shakespeare_data   = ShakespeareDataset(path=os.path.join(self.data_dir, self.data_name), chunk_size=200)
        shakespeare_loader = DataLoader(shakespeare_data, batch_size=32, shuffle=True)
        return shakespeare_loader, None

    def _train(self, train_loader: DataLoader) -> list[float]:
        model = self.model
        optimizer = self.optimizer

        criterion = nn.CrossEntropyLoss()
        i           : int   = 0       # ステップ数
        running_loss: float = 0.0     # 経験損失の合計
        tic         : float = time()  # 訓練開始の時間 (UNIX時間)

        for data in tqdm(train_loader):
            # inputsは初めから最後の手前の文字.
            inputs: torch.Tensor = data[:, :-1].to(self.device)
            # labelsは2文字目から最後の文字.
            labels: torch.Tensor = data[:, 1:].to(self.device)
            # print(labels.reshape(-1))

            outputs, _ = model(inputs)
            loss = criterion(outputs.reshape(-1, self._vocab_size), labels.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            i += 1
            
        train_loss: float = running_loss / i
        return [train_loss, time() - tic]
    
    def _eval(self, test_loader: DataLoader) -> list[float]:
        with torch.no_grad():
            print()
            print('+++++++++++++++++ generated +++++++++++++++++')
            print(generate_shakespeare(self.model, device=self.device))
            print('+++++++++++++++++++++++++++++++++++++++++++++')
        return []


def generate_shakespeare(
    net         : nn.Module,
    start_phrase: str='The King said',
    length      : int=200,
    device      : str='cpu'
) -> None:

    # モデルを評価モードにする.
    net.eval()
    # 出力の数値を格納するリスト.
    result = []

    # 開始文字列をTensorに変換.
    start_tensor: torch.Tensor = torch.tensor(
        _str2ints(start_phrase),
        dtype=torch.int64
    ).to(device)
    # 先頭にbatch次元を付ける.
    x0: torch.Tensor = start_tensor.unsqueeze(0)
    # RNNに通して出力と新しい内部状態を得る.
    o, h = net(x0)
    # 出力を正規化されていない確率に変換
    out_dist = o[:, -1].view(-1).exp()
    # 確率から実際の文字のインデックスをサンプリング.
    top_i = torch.multinomial(out_dist, 1)[0]

    # 生成された結果を次々にRNNに入力していく.
    for _ in range(length):
        inputs = torch.tensor([[top_i]], dtype=torch.int64)
        inputs = inputs.to(device)
        o, h = net(inputs, h)
        out_dist = o.view(-1).exp()
        top_i = torch.multinomial(out_dist, 1)[0]
        result.append(top_i)
    
    # 開始文字列と生成された文字列をまとめて返す.
    return start_phrase + _ints2str(result)
