import lightning as L

import torch


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = torch.randn(64, 32)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return 32


class RandomDataModule(L.LightningDataModule):
    def setup(self, stage: str) -> None:
        self.train_dataset = RandomDataset()
        self.validation_dataset = RandomDataset()

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=2, num_workers=8)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.validation_dataset, batch_size=2, num_workers=8)


class DummyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.layer = torch.nn.Linear(32, 2)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self.layer(batch).sum()
        self.log("training/loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.layer(batch).sum()
        self.log("validation/loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


def main():
    data_module = RandomDataModule()
    lightning_model = DummyModel()
    neptune_logger = L.pytorch.loggers.NeptuneLogger(project="p.pries/test", prefix="")
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        filename="model_{epoch:04d}",
        monitor="training/loss",
        mode="min",
        save_last=False,
        save_top_k=-1,
    )
    trainer = L.Trainer(
        max_epochs=-1,
        logger=neptune_logger,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        devices=[0],
        limit_train_batches=1,
        limit_val_batches=1,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(lightning_model, datamodule=data_module)


if __name__ == "__main__":
    main()
