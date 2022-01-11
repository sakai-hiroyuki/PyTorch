import torch
from torch import nn

__all__ = ['SequenceTaggingNet', 'SequenceGenerationNet']


class SequenceTaggingNet(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim : int   = 50,
        hidden_size   : int   = 50,
        num_layers    : int   = 1,
        dropout       : float = 0.2
    ) -> None:

        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, h0: torch.Tensor=None, l: torch.Tensor=None) -> torch.Tensor:
        # IDをEmbeddingで多次元のベクトルに変換する
        # xは(batch_size, step_size)
        # -> (batch_size, step_size, embedding_dim)
        x = self.emb(x)
        # 初期状態h0と共にRNNにxを渡す
        # xは(batch_size, step_size, embedding_dim)
        # -> (batch_size, step_size, hidden_dim)
        x, h = self.lstm(x, h0)
        # 最後のステップのみ取り出す
        # xは(batch_size, step_size, hidden_dim)
        # -> (batch_size, 1)
        if l is not None:
            # 入力のもともとの長さがある場合はそれを使用する
            x = x[list(range(len(x))), l-1, :]
        else:
            # なければ単純に最後を使用する
            x = x[:, -1, :]
        # 取り出した最後のステップを線形層に入れる
        x = self.linear(x)
        # 余分な次元を削除する
        # (batch_size, 1) -> (batch_size, )
        x = x.squeeze()
        return x


class SequenceGenerationNet(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int=50,
        hidden_size: int=50,
        num_layers: int=1,
        dropout=0.2
    ) -> None:

        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout
        )
        # Linearのoutputのサイズは最初のEmbeddingのinputサイズと同じnum_embeddings.
        self.linear = nn.Linear(hidden_size, num_embeddings)
    
    def forward(self, x: torch.Tensor, h0: torch.Tensor=None) -> torch.Tensor:
        x = self.emb(x)
        x, h = self.lstm(x, h0)
        x = self.linear(x)
        return x, h
