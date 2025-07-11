import torch
import torch.nn as nn
from tqdm import tqdm


def training(
        model,
        dataloader,
        lr=0.002,
        epochs=7,
        save=False,
        SAVEPATH="model.pt",
):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    print("start training")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"epoch: {epoch+1} / {epochs}"
        )

        for batch, (img, lbl) in progress_bar:
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, lbl)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            progress_bar.set_postfix(total_loss=f"{total_loss:2f}")
    
    if save:
        torch.save({"model": model.state_dict()}, SAVEPATH)
        print(f"saved model >> {SAVEPATH}")


def evaluation(
        model,
        dataloader
):
    print("evaluate model")
    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="evaluating"
        )
        nums = 0
        correct = 0
        for batch, (img, lbl) in progress_bar:
            pred = model(img)
            _, pred_labels = torch.max(pred.data, 1)
            nums += pred_labels.size(0)
            correct += (pred_labels == lbl).sum().item()
    
    acc = 100 * correct / nums
    return acc