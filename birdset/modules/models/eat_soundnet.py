import copy
from typing import List, Literal, Optional
from birdset.configs.model_configs import PretrainInfoConfig
from birdset.utils import pylogger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from dataclasses import dataclass, field
import warnings
import datasets
from typing import Tuple
from birdset.modules.models.birdset_model import BirdSetModel

log = pylogger.get_pylogger(__name__)


class ResBlock1dTF(nn.Module):
    def __init__(self, dim, dilation=1, kernel_size=3):
        super().__init__()
        self.block_t = nn.Sequential(
            nn.ReflectionPad1d(dilation * (kernel_size // 2)),
            nn.Conv1d(
                dim,
                dim,
                kernel_size=kernel_size,
                stride=1,
                bias=False,
                dilation=dilation,
                groups=dim,
            ),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2, True),
        )
        self.block_f = nn.Sequential(
            nn.Conv1d(dim, dim, 1, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2, True),
        )
        self.shortcut = nn.Conv1d(dim, dim, 1, 1)

    def forward(self, x):
        return self.shortcut(x) + self.block_f(x) + self.block_t(x)


class TAggregate(nn.Module):
    def __init__(
        self,
        clip_length=None,
        embedding_size=64,
        n_layers=6,
        nhead=6,
        n_classes=None,
        dim_feedforward=512,
        classifier: nn.Module | None = None,
    ):
        super(TAggregate, self).__init__()
        if classifier is None:
            classifier = nn.Linear(embedding_size, n_classes)
        self.fc = classifier

        self.num_tokens = 2  # TODO: changed this from 1 to 2
        drop_rate = 0.1
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=nhead,
            activation="gelu",
            dim_feedforward=dim_feedforward,
            dropout=drop_rate,
        )
        self.transformer_enc = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, norm=nn.LayerNorm(embedding_size)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_size))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, clip_length + self.num_tokens, embedding_size)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    # nn.init.constant_(m.weight, 1)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Parameter):
            with torch.no_grad():
                m.weight.data.normal_(0.0, 0.02)
                # nn.init.orthogonal_(m.weight)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        x.transpose_(1, 0)
        o = self.transformer_enc(x)
        pred = self.fc(o[0])
        return pred, o[0]


class AADownsample(nn.Module):
    def __init__(self, filt_size=3, stride=2, channels=None):
        super(AADownsample, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels
        ha = torch.arange(1, filt_size // 2 + 1 + 1, 1)
        a = torch.cat(
            (
                ha,
                ha.flip(
                    dims=[
                        -1,
                    ]
                )[1:],
            )
        ).float()
        a = a / a.sum()
        filt = a[None, :]
        self.register_buffer("filt", filt[None, :, :].repeat((self.channels, 1, 1)))

    def forward(self, x):
        x_pad = F.pad(x, (self.filt_size // 2, self.filt_size // 2), "reflect")
        y = F.conv1d(x_pad, self.filt, stride=self.stride, padding=0, groups=x.shape[1])
        return y


class Down(nn.Module):
    def __init__(self, channels, d=2, k=3):
        super().__init__()
        kk = d + 1
        self.down = nn.Sequential(
            nn.ReflectionPad1d(kk // 2),
            nn.Conv1d(channels, channels * 2, kernel_size=kk, stride=1, bias=False),
            nn.BatchNorm1d(channels * 2),
            nn.LeakyReLU(0.2, True),
            AADownsample(channels=channels * 2, stride=d, filt_size=k),
        )

    def forward(self, x):
        x = self.down(x)
        return x


class SoundNet(nn.Module):
    """
    NeuralNetwork for sound classification
    expected input shape: (batch_size, clip_length)
    output shape: (batch_size, num_classes)

    Parameters:
    nf (int): Number of filters in the convolutional layers. Default is 32.
    clip_length (Optional[int]): Length of the audio clips. Default is None.
    embedding_size (int): Dimension of the embeddings. Default is 128.
    n_layers (int): Number of transformer layers. Default is 4.
    nhead (int): Number of heads in the multihead attention models. Default is 8.
    factors (List[int]): List of factors for the convolutional layers. Default is [4, 4, 4, 4].
    num_classes (Optional[int]): Number of classes for classification. Default is None.
    dim_feedforward (int): Dimension of the feedforward network model. Default is 512.
    """

    def __init__(
        self,
        nf: int = 32,
        seq_len: int = 90112,
        embedding_size: int = 128,
        classifier: nn.Module | None = None,
        n_layers: int = 4,
        nhead: int = 8,
        factors: List[int] = [4, 4, 4, 4],
        dim_feedforward: int = 512,
        local_checkpoint: str | None = None,
        device: str = "cuda:0",
        num_classes: int | None = None,
    ):
        super().__init__()
        self.device = device

        ds_fac = np.prod(np.array(factors)) * 4
        clip_length = seq_len // ds_fac
        model = [
            nn.ReflectionPad1d(3),
            nn.Conv1d(1, nf, kernel_size=7, stride=1, bias=False),
            nn.BatchNorm1d(nf),
            nn.LeakyReLU(0.2, True),
        ]
        self.start = nn.Sequential(*model)
        model = []
        for i, f in enumerate(factors):
            model += [Down(channels=nf, d=f, k=f * 2 + 1)]
            nf *= 2
            if i % 2 == 0:
                model += [ResBlock1dTF(dim=nf, dilation=1, kernel_size=15)]
        self.down = nn.Sequential(*model)

        factors = [2, 2]
        model = []
        for _, f in enumerate(factors):
            for i in range(1):
                for j in range(3):
                    model += [ResBlock1dTF(dim=nf, dilation=3**j, kernel_size=15)]
            model += [Down(channels=nf, d=f, k=f * 2 + 1)]
            nf *= 2
        self.down2 = nn.Sequential(*model)
        self.project = nn.Conv1d(nf, embedding_size, 1)
        self.clip_length = clip_length
        self.tf = TAggregate(
            embedding_size=embedding_size,
            clip_length=clip_length,
            n_layers=n_layers,
            nhead=nhead,
            n_classes=num_classes,
            dim_feedforward=dim_feedforward,
            classifier=classifier,
        )
        self.apply(self._init_weights)
        if local_checkpoint:
            log.info(f">> Loading state dict from local checkpoint: {local_checkpoint}")
            self.start.load_state_dict(
                self.load_state_dict_from_file(local_checkpoint, model_name="start")
            )
            self.down.load_state_dict(
                self.load_state_dict_from_file(local_checkpoint, model_name="down")
            )
            self.down2.load_state_dict(
                self.load_state_dict_from_file(local_checkpoint, model_name="down2")
            )
            self.project.load_state_dict(
                self.load_state_dict_from_file(local_checkpoint, model_name="project")
            )
            self.tf.load_state_dict(
                self.load_state_dict_from_file(local_checkpoint, model_name="tf"),
                strict=False,
            )

    def load_state_dict_from_file(self, file_path, model_name="model"):
        state_dict = torch.load(file_path, map_location=self.device)["state_dict"]
        # select only models where the key starts with `model.` + model_name + `.`
        prefix = ""
        for name in state_dict.keys():
            if name.startswith("model.model."):
                prefix = "model."
                break
        # TODO: Only do this if classifier varies
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith(prefix+"model.tf.fc")
        }
        state_dict = {
            key: weight
            for key, weight in state_dict.items()
            if key.startswith(prefix+"model." + model_name + ".")
        }
        state_dict = {
            key.replace(prefix+"model." + model_name + ".", ""): weight
            for key, weight in state_dict.items()
        }

        return state_dict

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            with torch.no_grad():
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input_values: torch.Tensor, **kwargs):
        #!TODO: check shape for other models
        if len(input_values.shape) < 3:
            input_values.unsqueeze_(1)
        # has to be (batch x 1 x length)
        x = self.start(input_values)
        x = self.down(x)
        x = self.down2(x)
        x = self.project(x)
        pred, _ = self.tf(x)
        return pred

    # def get_embeddings(
    #     self, input_tensor: torch.Tensor
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     if len(input_tensor.shape) < 3:
    #         input_tensor = input_tensor.unsqueeze(1)

    #     # Pass through the initial layers
    #     x = self.start(input_tensor)
    #     x = self.down(x)
    #     x = self.down2(x)

    #     # Get the projected embeddings
    #     embeddings = self.project(x)

    #     # Create embeddings from transformer
    #     cls_tokens = self.tf.cls_token.expand(embeddings.shape[0], -1, -1)
    #     embeddings_with_pos = torch.cat(
    #         (cls_tokens, embeddings.permute(0, 2, 1).contiguous()), dim=1
    #     )
    #     embeddings_with_pos += self.tf.pos_embed
    #     embeddings_with_pos.transpose_(1, 0)
    #     transformer_output = self.tf.transformer_enc(embeddings_with_pos)

    #     # cls_embeddings are the transformer embeddings corresponding to the class token
    #     cls_embeddings = transformer_output[0]

    #     return cls_embeddings, None

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) < 3:
            x.unsqueeze_(1)
        x = self.start(x)
        x = self.down(x)
        x = self.down2(x)
        x = self.project(x)
        _, embeddings = self.tf(x)
        return embeddings


class EAT(BirdSetModel):
    EMBEDDING_SIZE = 128

    def __init__(
        self,
        num_classes: int | None,
        embedding_size: int = EMBEDDING_SIZE,
        local_checkpoint: str = None,
        load_classifier_checkpoint: bool = True,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = True,
        classifier: nn.Module | None = None,
        pretrain_info: Optional[PretrainInfoConfig] = None,
        device: str| int = "cuda:0",
        nf: int = 32,
        seq_len: int = 90112,
        n_layers: int = 4,
        nhead: int = 8,
        factors: List[int] = [4, 4, 4, 4],
        dim_feedforward: int = 512,
    ):
        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            local_checkpoint=local_checkpoint,
            load_classifier_checkpoint=load_classifier_checkpoint,
            freeze_backbone=freeze_backbone,
            preprocess_in_model=preprocess_in_model,
            pretrain_info=pretrain_info,
            classifier=classifier,
        )
        self.model = SoundNet(
            embedding_size=embedding_size,
            num_classes=self.num_classes,
            device=device,
            classifier=classifier,
            nf=nf,
            seq_len=seq_len,
            n_layers=n_layers,
            nhead=nhead,
            factors=factors,
            dim_feedforward=dim_feedforward,
            local_checkpoint=local_checkpoint,
        )

        if local_checkpoint and classifier is not None:
            if self.load_classifier_checkpoint:
                try:
                    state_dict = torch.load(self.local_checkpoint)["state_dict"]
                    classifier_state_dict = {
                        key.replace("model.classifier.", ""): weight
                        for key, weight in state_dict.items() if key.startswith("model.classifier.")
                    }
                    self.classifier.load_state_dict(classifier_state_dict, strict=False) # Strict to false in case BirdSet checkpoint is used without classifier weights
                except Exception as e:
                    log.error(f"Could not load classifier state dict from local checkpoint: {e}")  

        if self.freeze_backbone:
            self.classifier = copy.deepcopy(classifier)
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input_values: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.freeze_backbone:
            embedding = self.model.get_embeddings(input_values)
            logits = self.classifier(embedding)
            return logits
        return self.model(input_values)
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.get_embeddings(x)