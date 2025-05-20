from lightning.pytorch.callbacks import Callback, BaseFinetuning
from lightning.pytorch.trainer import Trainer

class Finetuning(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=3):
         super().__init__()
         self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, module):
        # freeze any module you want
        # Here, we are freezing `feature_extractor`
        self.freeze(module.model.model)
        
    def finetune_function(self, module, current_epoch, optimizer):
      if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=module.model.model,
                optimizer=optimizer,
                train_bn=True,
            )