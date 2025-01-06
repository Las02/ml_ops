from torch import nn
import torch
import data
import model
import typer
import seaborn as sns

app = typer.Typer()

@app.command()
def train(lr: float = 1e-3, n_epocs: int = 10) -> None:
    criterion = nn.NLLLoss()
    my_model = model.MyAwesomeModel()
    optimizer = torch.optim.Adam(my_model.parameters(), lr = 0.001)

    loss_all = list()
    epoc_all = list()

    for e in range(n_epocs):
        print(f"epoch {e}")
        train_loader, test_loader = data.Corrupt_mnist()
        for img, target in train_loader:

            optimizer.zero_grad()

            logits = my_model(img)
            loss : torch.Tensor = criterion(logits, target)
            loss.backward()

            optimizer.step()
            print(f"train loss: {loss}")

            loss_all.append(loss.item())
            epoc_all.append(e)

    fig  = sns.lineplot(x = epoc_all, y = loss_all).get_figure()
    fig.savefig("plots/sick_plot.png")


    torch.save(my_model.state_dict(), "./models/sick_model.pt")


@app.command()
def evaluate(model_checkpoint: str = "./models/sick_model.pt" ):
        train_loader, test_loader = data.Corrupt_mnist()
        my_model = model.MyAwesomeModel()
        my_model.load_state_dict(torch.load(model_checkpoint))
        criterion = nn.NLLLoss()

        # In this case there is only one entry in test_loader
        img, target = list(test_loader)[0]
        my_model.eval()
        logits: torch.Tensor = my_model(img)
        loss : torch.Tensor = criterion(logits, target)
        preds = logits.topk(1)[1]
        acc = sum(preds.flatten() ==  target) / len(target)
        print(f"test loss: {loss}, acc: {acc}")




if __name__ == "__main__":
    app()
