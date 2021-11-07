from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model.deep_emotion import DeepEmotionRecognitionModel
from helpers.data_generating_helper import GenerateData
from helpers.data_loading_helper import PlainDataset
import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(epochs, train_loader, val_loader, criterion, optimizer, device):
    for e in tqdm.trange(epochs):
        train_loss = validation_loss = train_correct = val_correct = 0

        # The training stage of the epoch
        net.train()
        for current_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)

        # The evaluation stage of the epoch: how well the NN is doing on new data
        net.eval()
        for current_idx, (data, labels) in enumerate(val_loader):
            data, labels = data.to(device), labels.to(device)
            val_outputs = net(data)
            val_loss = criterion(val_outputs, labels)
            validation_loss += val_loss.item()
            _, val_preds = torch.max(val_outputs, 1)
            val_correct += torch.sum(val_preds == labels.data)

        # Calculation of a current state of the loss function
        train_loss = train_loss/len(train_dataset)
        train_acc = float(train_correct) / len(train_dataset)
        validation_loss = validation_loss / len(validation_dataset)
        val_acc = float(val_correct) / len(validation_dataset)
        print(' Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f}'
              ' \tTraining Accuracy {:.3f}% \tValidation Accuracy {:.3f}%'
              .format(e+1, train_loss, validation_loss, train_acc*100, val_acc*100))

        if (epochs+1) % 10 == 0:
            torch.save(net.state_dict(), 'deep_emotion-{}-{}-{}-{}.pt'.format(epochs, batch, lr, val_acc))
            print("Network is saved to a file")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-s', '--setup', type=bool, help='setup the dataset for the first time')
    parser.add_argument('-d', '--data', type=str, required=True,
                        help='data folder that contains data files downloaded from Kaggle (train.csv and test.csv)')
    parser.add_argument('-hparams', '--hyperparams', type=bool,
                        help='True when changing the hyperparameters e.g (batch size, LR, num. of epochs)')
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs')
    parser.add_argument('-lr', '--learning_rate', type=float, help='value of learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, help='training/validation batch size')
    parser.add_argument('-t', '--train', type=bool, help='True when training')
    args = parser.parse_args()

    if args.setup:
        generate_dataset = GenerateData(args.data)
        generate_dataset.split_test()
        generate_dataset.save_images("train")
        generate_dataset.save_images('val')
        generate_dataset.save_images('test')

    if args.hyperparams:
        epochs = args.epochs     # Number of epochs of training
        lr = args.learning_rate  # The learning rate: how radically the network should adapt its weights
        batch = args.batch_size  # The size of a batch: number of pictures in one group (batch)
    else:
        epochs = 6
        lr = 0.005
        batch = 64

    if args.train:
        net = DeepEmotionRecognitionModel()
        net.to(device)
        print("Model architecture: ", net)
        train_csv_file = args.data+'/'+'train.csv'
        val_csv_file = args.data+'/'+'val.csv'
        train_img_dir = args.data+'/'+'train/'
        validation_img_dir = args.data+'/'+'val/'

        transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = PlainDataset(csv_file=train_csv_file, img_dir=train_img_dir,
                                     data_type='train', transform=transformation)
        validation_dataset = PlainDataset(csv_file=val_csv_file, img_dir=validation_img_dir,
                                          data_type='val', transform=transformation)

        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=0)
        val_loader = DataLoader(validation_dataset, batch_size=batch, shuffle=True, num_workers=0)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        # Initiate training
        train(epochs, train_loader, val_loader, criterion, optimizer, device)
