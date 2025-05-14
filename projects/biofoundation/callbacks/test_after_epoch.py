from lightning.pytorch.callbacks import Callback

class TestAfterEpoch(Callback):
    def on_validation_end(self, trainer, module):
        trainer.test(model=module, datamodule=trainer.datamodule)