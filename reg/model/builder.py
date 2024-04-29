from __future__ import annotations

from typing import Tuple
import os

import torch

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
        self.module = TransMorphModule()
        self.config = {}

    @classmethod
    def from_ckpt(cls, ckpt: str, strict: bool = False) -> TransMorphModuleBuilder:
        assert os.path.exists(ckpt), f"Path does not exist. Given: {ckpt}"
        assert os.path.isfile(ckpt), f"Path does not point to a file. Given: {ckpt}"
        builder = cls()
        builder.module = TransMorphModule.load_from_checkpoint(ckpt, strict=strict)
        builder.config = builder.module.config
        return builder

    def build(self) -> Tuple[TransMorphModule, dict]:
        self.module.config = self.config
        return self.module, self.config

    def set_network(self, config_identifier: str) -> TransMorphModuleBuilder:
        config = CONFIG_TM[config_identifier]
        descriptors = config_identifier.split("-")
        net = (
            TransMorphBayes(config)
            if len(descriptors) > 1 and descriptors[1] == "bayes"
            else TransMorph(config)
        )

        self.module.net = net
        self.config["network"] = config_identifier

        return self

    def set_criteria_warped(self, criteria_warped: str) -> TransMorphModuleBuilder:
        criteria = criteria_warped.split("-")
        criteria_warped = [
            (CONFIGS_WAPRED_LOSS[criteria[i]], float(criteria[i + 1]))
            for i in range(0, len(criteria), 2)
        ]

        self.module.criteria_warped = criteria_warped
        self.config["criteria_warped"] = [
            (loss_fn.__class__.__name__, w) for loss_fn, w in criteria_warped
        ]

        return self

    def set_criteria_flow(self, criteria_flow: str) -> TransMorphModuleBuilder:
        criteria = criteria_flow.split("-")
        criteria_flow = [
            (CONFIGS_FLOW_LOSS[criteria[i]], float(criteria[i + 1]))
            for i in range(0, len(criteria), 2)
        ]

        self.module.criteria_flow = criteria_flow
        self.config["criteria_flow"] = [
            (loss_fn.__class__.__name__, w) for loss_fn, w in criteria_flow
        ]

        return self

    def set_optimizer(self, optimizer: str) -> TransMorphModuleBuilder:
        optimizer = CONFIGS_OPTIMIZER[optimizer]

        self.module.optimizer = optimizer
        self.config["optimizer"] = optimizer.__name__

        return self

    def set_learning_rate(self, learning_rate: float) -> TransMorphModuleBuilder:
        self.module.learning_rate = learning_rate
        self.config["learning_rate"] = learning_rate
        return self

    def set_registration_strategy(
        self, registration_strategy: str
    ) -> TransMorphModuleBuilder:
        registration_strategy = RegistrationStrategy[registration_strategy.upper()]

        self.module.registration_strategy = registration_strategy
        self.config["registration_strategy"] = registration_strategy.name.lower()

        return self

    def set_registration_target(
        self, registration_target: str
    ) -> TransMorphModuleBuilder:
        registration_target = RegistrationTarget[registration_target.upper()]

        self.module.registration_target = registration_target
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

        self.module.net = net
        self.config["registration_depth"] = registration_depth

        return self

    def set_identity_loss(self, identity_loss: bool) -> TransMorphModuleBuilder:
        self.module.identity_loss = identity_loss
        self.config["identity_loss"] = identity_loss
        return self
