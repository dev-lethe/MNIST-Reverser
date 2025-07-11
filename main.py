import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import argparse

from model import MNIST_NN, MNIST_CNN
from train import training, evaluation
from generation import generation

def main(args):

    bs = args.bs
    lr = args.lr
    epochs = args.epochs
    target = args.target
    layer_dim = args.layer_dim
    ch = args.channel
    dim = args.dim
    data_dir = args.data_dir
    CONV = args.use_cnn
    LOAD = args.load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if CONV:
        model = MNIST_CNN(channel=ch, dim=dim).to(device)
        SAVENAME = "cnn_model.pt"
        save_dir = "cnn"
        print("use CNN model")
    else:
        model = MNIST_NN(layer_dim=layer_dim).to(device)
        SAVENAME = "nn_model.pt"
        save_dir = "nn"
        print("use NN model")

    model_path = os.path.join(data_dir, SAVENAME)
    if LOAD:
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            print(f"Loaded model from << {model_path}")
        else:
            print(f"Error: Model file not found at {model_path}.")
            print("Please train the model first by running without the --load_model flag.")
            return
    else:
        train_data_dir = os.path.join(data_dir, "train")
        train_data = torchvision.datasets.MNIST(root=train_data_dir, train=True, transform=transforms.ToTensor(), download=True)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=bs, shuffle=True)

        training(
            model=model,
            dataloader=train_dataloader,
            lr=lr,
            epochs=epochs,
            save=True,
            SAVEPATH=model_path
        )

        test_data_dir = os.path.join(data_dir, "test")
        test_data = torchvision.datasets.MNIST(root=test_data_dir, train=False, transform=transforms.ToTensor(), download=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=bs)

        acc = evaluation(
            model=model,
            dataloader=test_dataloader,
        )
        print(f"Model accuracy: {acc:.2f}%")

    print(f"Generating image for target: {target}")
    save_dir = os.path.join(data_dir, save_dir)
    generation(
        model,
        target=target,
        lr=args.gen_lr,
        lw=args.gen_lw,
        tvw=args.gen_tvw,
        biw=args.gen_biw,
        epochs=args.gen_epochs,
        conf=args.use_conf,
        CONV=CONV,
        save_dir=save_dir
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST Model Training and Generation CLI')

    parser.add_argument('--load_model', action='store_true', help='Load a pre-trained model instead of training.')
    parser.add_argument('--use_cnn', action='store_true', help='Use CNN model. If not specified, NN model is used.')

    parser.add_argument('--data_dir', type=str, default="./data", help='Directory for dataset and saved models.')

    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate for training.')
    parser.add_argument('--epochs', type=int, default=7, help='Number of training epochs.')
    parser.add_argument('--bs', type=int, default=64, help='Batch size.')

    parser.add_argument('--layer_dim', type=int, default=1024, help='Layer dimension for NN model.')
    parser.add_argument('--channel', '-ch', type=int, default=32, help='Base channel size for CNN model.')
    parser.add_argument('--dim', type=int, default=512, help='Hidden dimension for CNN model.')

    parser.add_argument('--target', type=int, default=0, help='Target digit for generation. (0~9)')
    parser.add_argument('--gen_lr', type=float, default=0.0005, help='Learning rate for generation.')
    parser.add_argument('--gen_lw', type=float, default=1, help='Loss weight for generation.')
    parser.add_argument('--gen_tvw', type=float, default=3e-4, help='Total Variation weight for generation.')
    parser.add_argument('--gen_biw', type=float, default=3e-4, help='Binarization weight for generation.')
    parser.add_argument('--gen_epochs', type=int, default=10000, help='Number of generation epochs.')
    parser.add_argument('--use_conf', action='store_true', help='Use confidence in generation loss.')

    args = parser.parse_args()
    main(args)
