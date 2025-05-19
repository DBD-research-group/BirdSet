from lightning.pytorch.callbacks import Callback

class TestAfterEpoch(Callback):
    def on_train_epoch_start(self, trainer, module):
        trainer.test(model=module, datamodule=trainer.datamodule)