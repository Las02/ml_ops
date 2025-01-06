from torch import nn
import torch
import data
import model
import typer

app = typer.Typer()

@app.command()
def train(lr: float = 1e-3, n_epocs: int = 10) -> None:
    criterion = nn.NLLLoss()
    my_model = model.MyAwesomeModel()
    optimizer = torch.optim.Adam(my_model.parameters(), lr = 0.001)

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
    torch.save(my_model.state_dict(), "./models/sick_model.pt")


@app.command()
def evaluate(model_checkpoint: str = "./models/sick_model.pt" ):
        train_loader, test_loader = data.Corrupt_mnist()
        my_model = model.MyAwesomeModel()
        my_model.load_state_dict(torch.load(model_checkpoint))
        criterion = nn.NLLLoss()

        for img, target in test_loader:
            my_model.eval()
            logits = my_model(img)
            loss : torch.Tensor = criterion(logits, target)
            print(f"test loss: {loss}")

if __name__ == "__main__":
    app()
