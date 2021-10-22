import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer=None, criterion=None, device=None):
        """Initialize the trainer"""
        self.model = model
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=.001)

        self.criterion = torch.nn.CrossEntropyLoss() if criterion is None else criterion

        if device is None:
            self.device = "cpu"
        else:
            self.device = device

        self.model = self.model.to(device)

    def get_model(self):
        return self.model

    def train(self, num_epochs, train_dataloader, val_dataloader=None):
        """Trains the model and logs the results"""
        # Set result dict
        results = {"train_loss": [], "train_acc": []}
        if val_dataloader is not None:
            results["val_loss"] = []
            results["val_acc"] = []

        # Start training
        for epoch in tqdm(range(num_epochs)):
            train_loss, train_acc = self.train_epoch(
                dataloader=train_dataloader)
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            # Validate only if we have a val dataloader
            if val_dataloader is not None:
                val_loss, val_acc = self.eval_epoch(dataloader=val_dataloader)
                results["val_loss"].append(val_loss)
                results["val_acc"].append(val_acc)

        return results

    def train_epoch(self, dataloader):
        """Trains one epoch"""
        self.model.train()
        total_loss = 0.
        total_correct = 0.
        for i, batch in enumerate(dataloader):
            # Send to device
            X, y = batch
            X = X.to(self.device)
            y = y.to(self.device)

            # Train step
            self.optimizer.zero_grad()  # Clear gradients.
            outs = self.model(X)  # Perform a single forward pass.
            loss = self.criterion(outs, y)

            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.

            # Compute metrics
            total_loss += loss.detach().item()
            total_correct += torch.sum(torch.argmax(outs,
                                       dim=-1) == y).detach().item()
        total_acc = total_correct / (len(dataloader) * dataloader.batch_size)
        return total_loss, total_acc

    def eval_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0.
        total_correct = 0.
        for i, batch in enumerate(dataloader):
            # Send to device
            X, y = batch
            X = X.to(self.device)
            y = y.to(self.device)

            # Eval
            outs = self.model(X)
            loss = self.criterion(outs, y)

            # Compute metrics
            total_loss += loss.detach().item()
            total_correct += torch.sum(torch.argmax(outs,
                                       dim=-1) == y).detach().item()
        total_acc = total_correct / (len(dataloader) * dataloader.batch_size)
        return total_loss, total_acc
