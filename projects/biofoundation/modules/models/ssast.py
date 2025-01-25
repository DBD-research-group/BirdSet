import torch.nn as nn
import torch
from timm.models.layers import trunc_normal_
import timm
from timm.models.layers import to_2tuple

from torchaudio.compliance import kaldi
import torch.nn.functional as F
from typing import Optional
from birdset.utils import pylogger

from biofoundation.modules.models.birdset_model import BirdSetModel

log = pylogger.get_pylogger(__name__)

#! This file includes code from SSAST by Yuan Gong, licensed under the BSD 3-Clause License
#! Copyright (c) 2022, Yuan Gong. All rights reserved.
#! Github-Repository: https://github.com/YuanGongND/ssast
#! Paper: https://ojs.aaai.org/index.php/AAAI/article/view/21315


# Custom PatchEmbed class
class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ASTModel(BirdSetModel):
    EMBEDDING_SIZE = 768

    def __init__(
        self,
        num_classes,
        embedding_size: int = EMBEDDING_SIZE,
        local_checkpoint: str = None,
        load_classifier_checkpoint: bool = True,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = True,
        classifier: nn.Module = None,
        fshape=16,
        tshape=16,
        fstride=16,
        tstride=16,
        input_fdim=128,
        input_tdim=500,
        model_size="base",
        pretrain_stage=False,
        load_pretrained_mdl_path=None,
    ):
        """
        Initialize a SSAST model for finetuning or evaluation. Pretraining is not supported but we need to keep parts of the code to initialize a SSAST model for finetuning/evaluation.

        Important Parameters:
        ---------------------
        1. fshape:
        - The side length of the patch along the frequency dimension.
        - Set to 16 as the paper uses 16x16 patches.
        - This value must remain consistent between pretraining and finetuning, so it cannot be changed unless using a different checkpoint.

        2. tshape:
        - The side length of the patch along the time dimension.
        - Similar to `fshape`, set to 16 for 16x16 patches.
        - Must also remain consistent between pretraining and finetuning.

        3. fstride:
        - The stride for patch splitting along the frequency dimension.
        - For 16x16 patches, `fstride=16` means no overlap, while `fstride=10` introduces an overlap of 6. (The paper uses 10)
        - Stride can differ between pretraining and finetuning, as pretraining doesn't use overlap. Smaller strides should improve performance but increase computational cost.

        4. tstride:
        - The stride for patch splitting along the time dimension.
        - Like `fstride`, `tstride=16` means no overlap, and `tstride=10` introduces an overlap of 6. (The paper uses 10)
        - The stride can be changed during finetuning to control overlap, with smaller strides improving performance but adding computational overhead. Pretraining uses no overlap.

        ! No improvements with smaller strides were noticed during testing

        5. input_fdim:
        - The number of frequency bins in the input spectrogram.
        - Typically 128 for mel-spectrograms.

        6. input_tdim:
        - The number of time frames in the input spectrogram.
        - Usually set to 500 for 5-second input spectrograms.

        Other Args:
        num_classes: The number of classes for a classifier head
        model_size: Must be the same as checkpoint model
        pretrain_stage: Always set to false (The model uses pretrain_stage=True to initialize the model for finetuning or evaluation)
        load_pretrained_mdl_path: Path of a checkpoint to load a pretrained model for finetuning or evaluation

        Configuration:
        --------------
        - Parameters that need adjustment for finetuning or evaluation can be modified in the `ssast.yaml` configuration file.
        """
        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            local_checkpoint=local_checkpoint,
            load_classifier_checkpoint=load_classifier_checkpoint,
            freeze_backbone=freeze_backbone,
            preprocess_in_model=preprocess_in_model,
        )

        if classifier is None:
            self.classifier = nn.Linear(embedding_size, num_classes)
        else:
            self.classifier = classifier

        # Prepare/load the model backbone
        assert (
            timm.__version__ == "0.4.5"
        ), "Please use timm == 0.4.5, the code might not be compatible with newer versions."
        # Override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        self.input_tdim = input_tdim  # For correct preprocessing()

        # pretrain_stage == True for creating the model backbone
        if pretrain_stage == True:
            if load_pretrained_mdl_path != None:
                raise ValueError(
                    "Setting load_pretrained_mdl_path at pretraining stage is useless, pretraining is always from scratch, please change it to None."
                )
            if fstride != fshape or tstride != tshape:
                raise ValueError(
                    "fstride != fshape or tstride != tshape, they must be same at the pretraining stage, patch split overlapping is not supported."
                )

            # There are different model sizes
            if model_size == "tiny":
                self.model = timm.create_model(
                    "vit_deit_tiny_distilled_patch16_224", pretrained=False
                )
                self.heads, self.depth = 3, 12
                self.cls_token_num = 2
            elif model_size == "small":
                self.model = timm.create_model(
                    "vit_deit_small_distilled_patch16_224", pretrained=False
                )
                self.heads, self.depth = 6, 12
                self.cls_token_num = 2
            elif model_size == "base":
                self.model = timm.create_model(
                    "vit_deit_base_distilled_patch16_384", pretrained=False
                )
                self.heads, self.depth = 12, 12
                self.cls_token_num = 2
            elif model_size == "base_nokd":
                self.model = timm.create_model(
                    "vit_deit_base_patch16_384", pretrained=False
                )
                self.heads, self.depth = 12, 12
                self.cls_token_num = 1
            else:
                raise Exception(
                    "Model size must be one of tiny, small, base, base_nokd"
                )

            self.original_embedding_dim = self.model.pos_embed.shape[2]

            # SSL Pretraining params
            self.fshape, self.tshape = fshape, tshape
            self.fstride, self.tstride = fstride, tstride
            self.input_fdim, self.input_tdim = input_fdim, input_tdim
            # this is a trick to make state_dict to track pretraining input_fdim and input_tdim and save them by using torch.save
            self.p_input_fdim, self.p_input_tdim = nn.Parameter(
                torch.tensor(input_fdim), requires_grad=False
            ), nn.Parameter(torch.tensor(input_tdim), requires_grad=False)

            # get the intermediate shape
            self.p_f_dim, self.p_t_dim = self.get_shape(
                fstride, tstride, input_fdim, input_tdim, fshape, tshape
            )
            num_patches = self.p_f_dim * self.p_t_dim
            self.num_patches = num_patches
            self.model.patch_embed.num_patches = num_patches

            # the linear patch projection layer, use 1 channel for spectrogram rather than the original 3 channels for RGB images.
            new_proj = torch.nn.Conv2d(
                1,
                self.original_embedding_dim,
                kernel_size=(fshape, tshape),
                stride=(fstride, tstride),
            )
            self.model.patch_embed.proj = new_proj

            # use trainable positional embedding
            new_pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    self.model.patch_embed.num_patches + self.cls_token_num,
                    self.original_embedding_dim,
                )
            )
            self.model.pos_embed = new_pos_embed
            trunc_normal_(self.model.pos_embed, std=0.02)

        # Use a pretrained models for finetuning
        elif pretrain_stage == False:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if load_pretrained_mdl_path == None:
                raise ValueError(
                    "Please set load_pretrained_mdl_path to load a pretrained model."
                )
            sd = torch.load(load_pretrained_mdl_path, map_location=device)
            # get the fshape and tshape, input_fdim and input_tdim in the pretraining stage
            try:
                p_fshape, p_tshape = (
                    sd["module.v.patch_embed.proj.weight"].shape[2],
                    sd["module.v.patch_embed.proj.weight"].shape[3],
                )
                p_input_fdim, p_input_tdim = (
                    sd["module.p_input_fdim"].item(),
                    sd["module.p_input_tdim"].item(),
                )
            except:
                raise ValueError(
                    "The model loaded is not from a torch.nn.Dataparallel object. Wrap it with torch.nn.Dataparallel and try again."
                )

            # during pretraining, fstride=fshape and tstride=tshape because no patch overlapping is used
            # here, input_fdim and input_tdim should be that used in pretraining, not that in the fine-tuning.
            # we need to know input_fdim and input_tdim to do positional embedding cut/interpolation.
            #! generally it should be better to use same input_fdim during pretraining and finetuning, but input_tdim can be safely different

            #! Here the same init is called again with pretrain_stage=True which is why we need parts of the code for pretraining
            audio_model = ASTModel(
                num_classes,
                fstride=p_fshape,
                tstride=p_tshape,
                fshape=p_fshape,
                tshape=p_tshape,
                input_fdim=p_input_fdim,
                input_tdim=p_input_tdim,
                pretrain_stage=True,
                model_size=model_size,
            )
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)

            self.model = audio_model.module.model
            self.original_embedding_dim = self.model.pos_embed.shape[2]
            self.cls_token_num = audio_model.module.cls_token_num

            # Classifier head for fine-tuning
            # self.mlp_head = nn.Sequential(
            # nn.LayerNorm(self.original_embedding_dim),
            # nn.Linear(self.original_embedding_dim, num_classes),
            # )

            f_dim, t_dim = self.get_shape(
                fstride, tstride, input_fdim, input_tdim, fshape, tshape
            )
            # Needed model changes for finetuning
            # patch array dimension during pretraining
            p_f_dim, p_t_dim = audio_model.module.p_f_dim, audio_model.module.p_t_dim
            num_patches = f_dim * t_dim
            p_num_patches = p_f_dim * p_t_dim
            self.model.patch_embed.num_patches = num_patches
            log.info(
                "Fine-tuning patch split stride: frequncey={:d}, time={:d}".format(
                    fstride, tstride
                )
            )
            log.info("Fine-tuning number of patches={:d}".format(num_patches))

            # patch shape should be same for pretraining and fine-tuning
            if fshape != p_fshape or tshape != p_tshape:
                raise ValueError(
                    "The patch shape of pretraining and fine-tuning is not consistant, pretraining: f={:d}, t={:d}, finetuning: f={:d}, t={:d}".format(
                        p_fshape, p_tshape, fshape, tshape
                    )
                )

            # patch split stride generally should be different for pretraining and fine-tuning, as patch split overlapping is only used in finetuning
            # during pretraining, p_fshape = p_fstride and p_tshape = p_tstride
            if fstride != p_fshape or tstride != p_tshape:
                # initialize a new patch embedding layer with desired new stride.
                new_proj = torch.nn.Conv2d(
                    1,
                    self.original_embedding_dim,
                    kernel_size=(fshape, tshape),
                    stride=(fstride, tstride),
                )
                # but the weights of patch embedding layer is still got from the pretrained models
                new_proj.weight = torch.nn.Parameter(
                    torch.sum(self.model.patch_embed.proj.weight, dim=1).unsqueeze(1)
                )
                new_proj.bias = self.model.patch_embed.proj.bias
                self.model.patch_embed.proj = new_proj

            new_pos_embed = (
                self.model.pos_embed[:, self.cls_token_num :, :]
                .detach()
                .reshape(1, p_num_patches, self.original_embedding_dim)
                .transpose(1, 2)
                .reshape(1, self.original_embedding_dim, p_f_dim, p_t_dim)
            )
            # cut or interpolate the positional embedding
            if t_dim < p_t_dim:
                new_pos_embed = new_pos_embed[
                    :,
                    :,
                    :,
                    int(p_t_dim / 2)
                    - int(t_dim / 2) : int(p_t_dim / 2)
                    - int(t_dim / 2)
                    + t_dim,
                ]
            else:
                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(8, t_dim), mode="bilinear"
                )
            if f_dim < p_f_dim:
                new_pos_embed = new_pos_embed[
                    :,
                    :,
                    int(p_f_dim / 2)
                    - int(f_dim / 2) : int(p_f_dim / 2)
                    - int(f_dim / 2)
                    + t_dim,
                    :,
                ]
            else:
                new_pos_embed = torch.nn.functional.interpolate(
                    new_pos_embed, size=(f_dim, t_dim), mode="bilinear"
                )
            # Error here
            new_pos_embed = new_pos_embed.reshape(
                1, self.original_embedding_dim, num_patches
            ).transpose(1, 2)
            self.model.pos_embed = nn.Parameter(
                torch.cat(
                    [
                        self.model.pos_embed[:, : self.cls_token_num, :].detach(),
                        new_pos_embed,
                    ],
                    dim=1,
                )
            )

            if local_checkpoint:
                self._load_local_checkpoint()

            if freeze_backbone:
                for param in self.model.parameters():
                    param.requires_grad = False

    # Get the shape of intermediate representation.
    def get_shape(self, fstride, tstride, input_fdim, input_tdim, fshape, tshape):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(
            1,
            self.original_embedding_dim,
            kernel_size=(fshape, tshape),
            stride=(fstride, tstride),
        )
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def preprocess(
        self, input_values: torch.Tensor, input_tdim=500, sampling_rate=16000
    ) -> torch.Tensor:
        """
        Preprocesses the input values by applying mel-filterbank transformation. Same as for AudioMae and ConvNeXt.
        Args:
            input_values (torch.Tensor): Input tensor of shape (batch_size, num_samples).
            input_tdim (int): The number of frames to keep. Defaults to 500.
            sampling_rate (int): The sampling rate of the input tensor. Defaults to 16000.
        Returns:
            torch.Tensor: Preprocessed tensor of shape (batch_size, 1, num_mel_bins, num_frames).
        """
        device = input_values.device
        melspecs = []
        for waveform in input_values:
            melspec = kaldi.fbank(
                waveform,
                htk_compat=True,
                window_type="hanning",
                num_mel_bins=128,
                use_energy=False,
                sample_frequency=sampling_rate,
                frame_shift=10,
            )  # shape (n_frames, 128)
            if melspec.shape[0] < input_tdim:
                melspec = F.pad(melspec, (0, 0, 0, input_tdim - melspec.shape[0]))
            else:
                melspec = melspec[:input_tdim]
            melspecs.append(melspec)
        melspecs = torch.stack(melspecs).to(device)
        melspecs = melspecs.unsqueeze(1)  # shape (batch_size, 1, 128, 1024)
        # melspecs = (melspecs - self.MEAN) / (self.STD * 2)
        return melspecs

    def get_embeddings(self, input_tensor) -> torch.Tensor:
        """
        Calculates embeddings for the given input tensor which is first converted to frequency bins.
        Args:
            input_tensor (torch.Tensor): The input tensor of shape (batch_size, 1, waveform).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the embeddings tensor and the output of the MLP head.
        """
        if self.preprocess_in_model:
            input_tensor = self.preprocess(
                input_tensor, input_tdim=self.input_tdim
            )  # Convert to spectrogram
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        input_tensor = input_tensor.transpose(2, 3)
        B = input_tensor.shape[0]
        x = self.model.patch_embed(input_tensor)

        if self.cls_token_num == 2:
            cls_tokens = self.model.cls_token.expand(B, -1, -1)
            dist_token = self.model.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.model.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        for blk_id, blk in enumerate(self.model.blocks):
            x = blk(x)

        x = self.model.norm(x)
        # average output of all tokens except cls token(s)
        x = torch.mean(x[:, self.cls_token_num :, :], dim=1)

        return x

    def forward(
        self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        embeddings = self.get_embeddings(input_values)

        return self.classifier(embeddings)
