import string

from torch import nn
from torch.optim import Adam

from models import (
    resnet20_cifar10,
    SimpleCNN_MNIST,
    SequenceTaggingNet,
    SequenceGenerationNet
)
from experiments import (
    ExperimentCIFAR10,
    ExperimentMNIST,
    ExperimentIMDb,
    ExperimentShakespeare
)

from torchsummary import summary


def cifar10() -> nn.Module:
    model = resnet20_cifar10()
    optimizer = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    
    summary(model, [3, 32, 32])

    experiment = ExperimentCIFAR10(
        model      = model,
        optimizer  = optimizer,
        max_epoch  = 2,
        batch_size = 256,
        csv_dir    = './results/csv/cifar10/resnet20',
        csv_name   = 'adam.csv',
        prm_dir    = './results/prm/cifar10/resnet20',
        prm_name   = 'adam.prm',
        cutout     = True
    )

    return experiment()


def mnist() -> nn.Module:
    model = SimpleCNN_MNIST()
    optimizer = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

    summary(model, [1, 28, 28])

    experiment = ExperimentMNIST(
        model      = model,
        optimizer  = optimizer,
        max_epoch  = 2,
        batch_size = 256,
        csv_dir    = './results/csv/mnist/cnn',
        csv_name   = 'adam.csv',
        prm_dir    = './results/prm/mnist/cnn',
        prm_name   = 'adam.prm'
    )

    return experiment()


def imdb() -> nn.Module:
    model = SequenceTaggingNet(num_embeddings=89528, num_layers=2)
    optimizer = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

    summary(model)

    experiment = ExperimentIMDb(
        model      = model,
        optimizer  = optimizer,
        max_epoch  = 2,
        batch_size = 32,
        csv_dir    = './results/csv/imdb/lstm',
        csv_name   = 'adam.csv',
        prm_dir    = './results/prm/imdb/lstm',
        prm_name   = 'adam.prm',
    )

    return experiment()


def shakespeare() -> nn.Module:
    model = SequenceGenerationNet(
        num_embeddings = len(string.printable),
        embedding_dim  = 20,
        hidden_size    = 50,
        num_layers     = 2,
        dropout        = 0.2
    )
    optimizer = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

    summary(model)

    experiment = ExperimentShakespeare(
        model      = model,
        optimizer  = optimizer,
        max_epoch  = 10,
        batch_size = 32,
        csv_dir    = './results/csv/shakespeare/lstm',
        csv_name   = 'adam.csv',
        prm_dir    = './results/prm/shakespeare/lstm',
        prm_name   = 'adam.prm'
    )

    return experiment()


if __name__ == '__main__':
    shakespeare()
    