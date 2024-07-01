"""Example model file for track 2."""

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import resize
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from anomalib.models.image.winclip.torch_model import WinClipModel
from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel, EfficientAdModelSize
from anomalib.models.image.efficient_ad.lightning_model import EfficientAd
from anomalib.data import AnomalibDataModule

from glob import glob
import numpy as np
from os.path import expanduser, join, exists, basename, splitext
from os import mkdir

from typing import Any, Mapping
import time


class PretrainEfficientAD(nn.Module):
    """Example model class for track 2.

    This class applies few-shot anomaly detection using the WinClip model from Anomalib.
    """

    def __init__(self) -> None:
        super().__init__()

        self.EfficientAD = EfficientAd()
        self.register_buffer("_map_st", torch.empty(0))
        self.register_buffer("_map_ae", torch.empty(0))
        self.register_buffer("_ref_score", torch.empty(0))
        # NOTE: Create your transformation pipeline (if needed).
        self.EfficientAD._transform = self.EfficientAD.configure_transforms(
            image_size=(256, 256))
        assert self.EfficientAD._transform != None, "ERROR:: Transform should not be None."

        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.EfficientAD.to(device)
        self.EfficientAD.eval()

        self.k_shot = None

    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Transform the input batch and pass it through the model.

        This model returns a dictionary with the following keys
        - ``anomaly_map`` - Anomaly map.
        - ``pred_score`` - Predicted anomaly score.
        """

        batch = self.EfficientAD._transform(batch)
        output = self.EfficientAD(batch)
        anomaly_maps, map_st, map_ae = output["anomaly_map"], output["map_st"], output["map_ae"]
        # resize back to 256x256 for evaluation
        anomaly_maps = resize(anomaly_maps, (256, 256))
        pred_score = anomaly_maps.mean()

        if self.k_shot is not None:
            few_shot_scores = 1-self._compute_few_shot_scores(map_st, map_ae)
            # pred_score = pred_score / self._ref_score
            pred_score = pred_score * few_shot_scores
            # anomaly_maps = (anomaly_maps + few_shot_scores) / 2
            # image_scores = (image_scores + few_shot_scores.amax(dim=(-2, -1))) / 2

        return {"pred_score": pred_score, "anomaly_map": anomaly_maps}

    def fit(self,
            cur_epoch: int,
            datamodule: AnomalibDataModule,
            max_epochs: int = 20,
            max_steps: int = 70000,
            # save_per_epochs: int = 50,
            ckpt_dir: str | None = None,
            ckpt_path: str | None = None,
            ) -> None:
        start_time = time.time()
        self.train()
        self.EfficientAD.train()
        # checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
        #                                       filename='{cur_epoch}-{step}',
        #                                       save_last=True,
        #                                     #   every_n_epochs= save_per_epochs,
        #                                       verbose=True)
        checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                              filename='last',
                                              save_last=False,
                                              # save_top_k= 1,
                                              verbose=True,
                                              # monitor='val_loss', # or another relevant metric
                                              # mode='min'
                                              )
        self.trainer = Trainer(max_epochs=max_epochs,
                               max_steps=max_steps,
                               check_val_every_n_epoch=max_epochs,
                               enable_checkpointing=True,
                               callbacks=[checkpoint_callback],
                               default_root_dir=ckpt_dir)
        self.trainer.fit(self.EfficientAD,
                         datamodule=datamodule, ckpt_path=ckpt_path)
        # self.trainer.save_checkpoint(ckpt_dir+f'epoch_{cur_epoch}_{datamodule.category}_weights.ckpt', weights_only=True)
        if (cur_epoch % 100 == 0):
            torch.save(self.EfficientAD.state_dict(), ckpt_dir +
                       f'epoch_{cur_epoch}_{datamodule.category}_weights.pth')

        with open("pretrain_process.txt", "a") as f:
            end_time = time.time()
            exe_time = time.strftime(
                '%H:%M:%S', time.localtime(end_time - start_time))
            f.write(
                f"Epoch {cur_epoch} - Time : {exe_time} - Category : {datamodule.category}\n")
            f.close()

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        # super().load_state_dict(state_dict, strict, assign)
        # self.EfficientAD = EfficientAd.load_from_checkpoint(state_dict)
        self.EfficientAD.load_state_dict(state_dict)

    def setup(self, data: dict) -> None:
        """Setup the few-shot samples for the model.

        The evaluation script will call this method to pass the k images for few shot learning and the object class
        name. In the case of MVTec LOCO this will be the dataset category name (e.g. breakfast_box). Please contact
        the organizing committee if if your model requires any additional dataset-related information at setup-time.
        """
        few_shot_samples = data.get("few_shot_samples")
        # class_name = data.get("dataset_category")

        few_shot_samples = self.EfficientAD._transform(few_shot_samples)
        # self.model.setup(class_name, few_shot_samples)
        self._setup(few_shot_samples)

    def _setup(self, reference_images: torch.Tensor | None = None) -> None:
        self.reference_images = reference_images if reference_images is not None else self.reference_images
        if self.reference_images is not None:
            # update k_shot based on number of reference images
            self.k_shot = self.reference_images.shape[0]
            self._collect_visual_embeddings(self.reference_images)

    @torch.no_grad
    def _collect_visual_embeddings(self, images: torch.Tensor) -> None:
        """Collect visual embeddings based on a set of normal reference images.

        Args:
            images (torch.Tensor): Tensor of shape ``(K, C, H, W)`` containing the reference images.
        """
        output = self.EfficientAD(images)
        reference_map, self._map_st, self._map_ae = output[
            "anomaly_map"], output["map_st"], output["map_ae"]
        self._ref_score = reference_map.mean()
        if self._map_st.ndim >= 2:
            self._map_st = self._map_st.view(self.k_shot, -1)
        if self._map_ae.ndim >= 2:
            self._map_ae = self._map_ae.view(self.k_shot, -1)

    def _compute_few_shot_scores(self, map_st: torch.Tensor, map_ae: torch.Tensor) -> torch.Tensor:
        batch_size = map_st.shape[0]
        st_score = F.cosine_similarity(
            map_st.view(batch_size, -1), self._map_st)
        ae_score = F.cosine_similarity(
            map_ae.view(batch_size, -1), self._map_ae)
        return (st_score + ae_score).mean()


class EfficientAD(nn.Module):
    """Example model class for track 2.

    This class applies few-shot anomaly detection using the WinClip model from Anomalib.
    """

    def __init__(self) -> None:
        super().__init__()

        self.EfficientAD = EfficientAd()
        self.register_buffer("_map_st", torch.empty(0))
        self.register_buffer("_map_ae", torch.empty(0))
        self.register_buffer("_ref_score", torch.empty(0))
        # NOTE: Create your transformation pipeline (if needed).
        self.EfficientAD._transform = self.EfficientAD.configure_transforms(
            image_size=(256, 256))
        assert self.EfficientAD._transform != None, "ERROR:: Transform should not be None."

        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.EfficientAD.to(device)
        self.EfficientAD.eval()

        self.k_shot = None

    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Transform the input batch and pass it through the model.

        This model returns a dictionary with the following keys
        - ``anomaly_map`` - Anomaly map.
        - ``pred_score`` - Predicted anomaly score.
        """

        batch = self.EfficientAD._transform(batch)
        output = self.EfficientAD(batch)
        anomaly_maps, map_st, map_ae = output["anomaly_map"], output["map_st"], output["map_ae"]
        # resize back to 256x256 for evaluation
        anomaly_maps = resize(anomaly_maps, (256, 256))
        pred_score = anomaly_maps.mean()

        return {"pred_score": pred_score, "anomaly_map": anomaly_maps}

    def fit(self,
            cur_epoch: int,
            datamodule: AnomalibDataModule,
            max_epochs: int = 20,
            max_steps: int = 70000,
            # save_per_epochs: int = 50,
            ckpt_dir: str | None = None,
            ckpt_path: str | None = None,
            ) -> None:
        start_time = time.time()
        self.train()
        self.EfficientAD.train()
        # checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
        #                                       filename='{cur_epoch}-{step}',
        #                                       save_last=True,
        #                                     #   every_n_epochs= save_per_epochs,
        #                                       verbose=True)
        checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                              filename='last',
                                              save_last=False,
                                              # save_top_k= 1,
                                              verbose=True,
                                              # monitor='val_loss', # or another relevant metric
                                              # mode='min'
                                              )
        self.trainer = Trainer(max_epochs=max_epochs,
                               max_steps=max_steps,
                               check_val_every_n_epoch=max_epochs,
                               enable_checkpointing=True,
                               callbacks=[checkpoint_callback],
                               default_root_dir=ckpt_dir)
        self.trainer.fit(self.EfficientAD,
                         datamodule=datamodule, ckpt_path=ckpt_path)
        # self.trainer.save_checkpoint(ckpt_dir+f'epoch_{cur_epoch}_{datamodule.category}_weights.ckpt', weights_only=True)
        if (cur_epoch % 100 == 0):
            torch.save(self.EfficientAD.state_dict(), ckpt_dir +
                       f'epoch_{cur_epoch}_{datamodule.category}_weights.pth')

        with open("pretrain_process.txt", "a") as f:
            end_time = time.time()
            exe_time = time.strftime(
                '%H:%M:%S', time.localtime(end_time - start_time))
            f.write(
                f"Epoch {cur_epoch} - Time : {exe_time} - Category : {datamodule.category}\n")
            f.close()

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        # super().load_state_dict(state_dict, strict, assign)
        # self.EfficientAD = EfficientAd.load_from_checkpoint(state_dict)
        self.EfficientAD.load_state_dict(state_dict)

    def setup(self, data: dict) -> None:
        """Setup the few-shot samples for the model.

        The evaluation script will call this method to pass the k images for few shot learning and the object class
        name. In the case of MVTec LOCO this will be the dataset category name (e.g. breakfast_box). Please contact
        the organizing committee if if your model requires any additional dataset-related information at setup-time.
        """
        few_shot_samples = data.get("few_shot_samples")
        # class_name = data.get("dataset_category")

        few_shot_samples = self.EfficientAD._transform(few_shot_samples)
        # self.model.setup(class_name, few_shot_samples)
        self._setup(few_shot_samples)

    def _setup(self, reference_images: torch.Tensor | None = None) -> None:
        self.reference_images = reference_images if reference_images is not None else self.reference_images
        if self.reference_images is not None:
            # update k_shot based on number of reference images
            self.k_shot = self.reference_images.shape[0]
            self._collect_visual_embeddings(self.reference_images)

    @torch.no_grad
    def _collect_visual_embeddings(self, images: torch.Tensor) -> None:
        """Collect visual embeddings based on a set of normal reference images.

        Args:
            images (torch.Tensor): Tensor of shape ``(K, C, H, W)`` containing the reference images.
        """
        output = self.EfficientAD(images)
        reference_map, self._map_st, self._map_ae = output[
            "anomaly_map"], output["map_st"], output["map_ae"]
        self._ref_score = reference_map.mean()
        if self._map_st.ndim >= 2:
            self._map_st = self._map_st.view(self.k_shot, -1)
        if self._map_ae.ndim >= 2:
            self._map_ae = self._map_ae.view(self.k_shot, -1)

    def _compute_few_shot_scores(self, map_st: torch.Tensor, map_ae: torch.Tensor) -> torch.Tensor:
        batch_size = map_st.shape[0]
        st_score = F.cosine_similarity(
            map_st.view(batch_size, -1), self._map_st)
        ae_score = F.cosine_similarity(
            map_ae.view(batch_size, -1), self._map_ae)
        return (st_score + ae_score).mean()
