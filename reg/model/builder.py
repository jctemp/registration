from __future__ import annotations

from typing import Tuple, Any
import os

from pathlib2 import Path
import torch
import json

from reg.transmorph.transmorph_bayes import TransMorphBayes
from reg.transmorph.transmorph import TransMorph
from reg.transmorph.configs import CONFIG_TM
from reg.metrics import CONFIGS_WAPRED_LOSS, CONFIGS_FLOW_LOSS
from reg.model.model import TransMorphModule, RegistrationStrategy, RegistrationTarget

CONFIGS_OPTIMIZER = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adam-w": torch.optim.AdamW,
}


class TransMorphModuleBuilder:
    def __init__(self):
        self.model = None
        self.config = {}
        self.hyperparams = {
            "net": TransMorph(CONFIG_TM["transmorph"]),
            "criteria_warped": [(CONFIGS_WAPRED_LOSS["gmi"], 1.0)],
            "criteria_flow": [(CONFIGS_FLOW_LOSS["gl2d"], 1.0)],
            "registration_target": RegistrationTarget.LAST,
            "registration_strategy": RegistrationStrategy.SOREG,
            "registration_depth": 32,
            "registration_stride": 1,
            "identity_loss": False,
            "optimizer": torch.optim.Adam,
            "learning_rate": 1e-4,
        }

    @classmethod
    def from_ckpt(cls, ckpt: Any, strict: bool = False) -> TransMorphModuleBuilder:
        assert os.path.exists(ckpt), f"Path does not exist. Given: {ckpt}"
        assert os.path.isfile(ckpt), f"Path does not point to a file. Given: {ckpt}"

        builder = cls()

        builder.model = TransMorphModule.load_from_checkpoint(str(ckpt), strict=strict)
        builder.hyperparams["net"] = builder.model.net
        builder.hyperparams["criteria_warped"] = builder.model.criteria_warped
        builder.hyperparams["criteria_flow"] = builder.model.criteria_flow
        builder.hyperparams["registration_target"] = builder.model.registration_target
        builder.hyperparams[
            "registration_strategy"
        ] = builder.model.registration_strategy
        builder.hyperparams["registration_depth"] = builder.model.registration_depth
        builder.hyperparams["registration_stride"] = builder.model.registration_stride
        builder.hyperparams["identity_loss"] = builder.model.identity_loss
        builder.hyperparams["optimizer"] = builder.model.optimizer
        builder.hyperparams["learning_rate"] = builder.model.learning_rate
        builder.config = builder.model.config

        return builder

    def build(self) -> Tuple[TransMorphModule, dict]:
        if self.model is None:
            self.model = TransMorphModule(
                net=self.hyperparams["net"],
                criteria_warped=self.hyperparams["criteria_warped"],
                criteria_flow=self.hyperparams["criteria_flow"],
                registration_target=self.hyperparams["registration_target"],
                registration_strategy=self.hyperparams["registration_strategy"],
                registration_depth=self.hyperparams["registration_depth"],
                registration_stride=self.hyperparams["registration_stride"],
                identity_loss=self.hyperparams["identity_loss"],
                optimizer=self.hyperparams["optimizer"],
                learning_rate=self.hyperparams["learning_rate"],
                config=self.config,
            )
        else:
            self.model.net = self.hyperparams["net"]
            self.model.criteria_warped = (self.hyperparams["criteria_warped"],)
            self.model.criteria_flow = (self.hyperparams["criteria_flow"],)
            self.model.registration_target = (self.hyperparams["registration_target"],)
            self.model.registration_strategy = (
                self.hyperparams["registration_strategy"],
            )
            self.model.registration_depth = (self.hyperparams["registration_depth"],)
            self.model.registration_stride = (self.hyperparams["registration_stride"],)
            self.model.identity_loss = (self.hyperparams["identity_loss"],)
            self.model.optimizer = (self.hyperparams["optimizer"],)
            self.model.learning_rate = (self.hyperparams["learning_rate"],)
            self.model.config = self.config

        return self.model, self.config

    def set_network(self, config_identifier: str) -> TransMorphModuleBuilder:
        config = CONFIG_TM[config_identifier]
        config.img_size = (*config.img_size[:-1], self.hyperparams["registration_depth"])
        descriptors = config_identifier.split("-")

        net = (
            TransMorphBayes(config)
            if len(descriptors) > 1 and descriptors[1] == "bayes"
            else TransMorph(config)
        )

        self.hyperparams["net"] = net
        self.config["network"] = config_identifier

        return self

    def set_criteria_warped(
        self, criteria_warped: str | list[tuple[str, float]]
    ) -> TransMorphModuleBuilder:
        if isinstance(criteria_warped, str):
            criteria = criteria_warped.split("-")
        else:
            criteria = [value for entry in criteria_warped for value in entry]

        criteria_warped = [
            (criteria[i], CONFIGS_WAPRED_LOSS[criteria[i]], float(criteria[i + 1]))
            for i in range(0, len(criteria), 2)
        ]

        self.hyperparams["criteria_warped"] = [(loss_fn, w) for (_, loss_fn, w) in criteria_warped]
        self.config["criteria_warped"] = [
            (name, w) for (name, loss_fn, w) in criteria_warped
        ]

        return self

    def set_criteria_flow(
        self, criteria_flow: str | list[tuple[str, float]]
    ) -> TransMorphModuleBuilder:
        if isinstance(criteria_flow, str):
            criteria = criteria_flow.split("-")
        else:
            criteria = [value for entry in criteria_flow for value in entry]

        criteria_flow = [
            (criteria[i], CONFIGS_FLOW_LOSS[criteria[i]], float(criteria[i + 1]))
            for i in range(0, len(criteria), 2)
        ]

        self.hyperparams["criteria_flow"] = [(loss_fn, w) for (_, loss_fn, w) in criteria_flow]
        self.config["criteria_flow"] = [
            (name, w) for (name, loss_fn, w) in criteria_flow
        ]

        return self

    def set_registration_strategy(
        self, registration_strategy: str
    ) -> TransMorphModuleBuilder:
        registration_strategy = RegistrationStrategy[registration_strategy.upper()]

        self.hyperparams["registration_strategy"] = registration_strategy
        self.config["registration_strategy"] = registration_strategy.name.lower()

        return self

    def set_registration_target(
        self, registration_target: str
    ) -> TransMorphModuleBuilder:
        registration_target = RegistrationTarget[registration_target.upper()]

        self.hyperparams["registration_target"] = registration_target
        self.config["registration_target"] = registration_target.name.lower()

        return self

    def set_registration_depth(
        self, registration_depth: int
    ) -> TransMorphModuleBuilder:
        config_identifier = self.config["network"]

        config = CONFIG_TM[config_identifier]
        config.img_size = (*config.img_size[:-1], registration_depth)

        descriptors = config_identifier.split("-")
        net = (
            TransMorphBayes(config)
            if len(descriptors) > 1 and descriptors[1] == "bayes"
            else TransMorph(config)
        )

        self.model.net = net
        self.hyperparams["net"] = net
        self.hyperparams["registration_depth"] = registration_depth
        self.config["registration_depth"] = registration_depth

        return self

    def set_registration_stride(
        self, registration_stride: int
    ) -> TransMorphModuleBuilder:
        self.hyperparams["registration_stride"] = registration_stride
        self.config["registration_stride"] = registration_stride
        return self

    def set_identity_loss(self, identity_loss: bool) -> TransMorphModuleBuilder:
        self.hyperparams["identity_loss"] = identity_loss
        self.config["identity_loss"] = identity_loss
        return self

    def set_optimizer(self, optimizer: str) -> TransMorphModuleBuilder:
        optimizer_name = optimizer
        optimizer = CONFIGS_OPTIMIZER[optimizer]

        self.hyperparams["optimizer"] = optimizer
        self.config["optimizer"] = optimizer_name

        return self

    def set_learning_rate(self, learning_rate: float) -> TransMorphModuleBuilder:
        self.hyperparams["learning_rate"] = learning_rate
        self.config["learning_rate"] = learning_rate
        return self
