from __future__ import annotations

from typing import Tuple, Any
import os

from reg.wrapper.model import CONFIGS_OPTIMIZER, TransMorphModule, RegistrationTarget, RegistrationStrategy
from reg.measure import CONFIGS_WAPRED_LOSS, CONFIGS_FLOW_LOSS


class TransMorphModuleBuilder:
    def __init__(self):
        self.is_ckpt = False
        self.model = None
        self.hyperparams = {
            "network": "transmorph",
            "criteria_warped": [("mse", 1.0)],
            "criteria_flow": [("gl2d", 1.0)],
            "registration_target": "last",
            "registration_strategy": "soreg",
            "registration_depth": 32,
            "registration_stride": 1,
            "identity_loss": False,
            "optimizer": "adam",
            "learning_rate": 1e-4,
        }

    @classmethod
    def from_ckpt(cls, ckpt: Any, strict: bool = False) -> TransMorphModuleBuilder:
        assert os.path.exists(ckpt), f"Path does not exist. Given: {ckpt}"
        assert os.path.isfile(ckpt), f"Path does not point to a file. Given: {ckpt}"

        builder = cls()
        builder.is_ckpt = True
        builder.model = TransMorphModule.load_from_checkpoint(str(ckpt), strict=strict)

        builder.hyperparams["network"] = builder.model.network
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

        return builder

    def build(self) -> Tuple[TransMorphModule, dict]:
        if self.model is None:
            self.model = TransMorphModule(
                network=self.hyperparams["network"],
                criteria_warped=self.hyperparams["criteria_warped"],
                criteria_flow=self.hyperparams["criteria_flow"],
                registration_target=self.hyperparams["registration_target"],
                registration_strategy=self.hyperparams["registration_strategy"],
                registration_depth=self.hyperparams["registration_depth"],
                registration_stride=self.hyperparams["registration_stride"],
                identity_loss=self.hyperparams["identity_loss"],
                optimizer=self.hyperparams["optimizer"],
                learning_rate=self.hyperparams["learning_rate"],
            )
        else:
            self.model.registration_stride = self.hyperparams["registration_stride"]
            self.model.identity_loss = self.hyperparams["identity_loss"]
            self.model.learning_rate = self.hyperparams["learning_rate"]

            self.model.criteria_warped = self.hyperparams["criteria_warped"]
            self.model.criteria_flow = self.hyperparams["criteria_flow"]
            self.model.registration_target = self.hyperparams["registration_target"]
            self.model.registration_strategy = self.hyperparams["registration_strategy"]
            self.model.optimizer = self.hyperparams["optimizer"]

            self.model.criteria_warped_nnf = [
                (CONFIGS_WAPRED_LOSS[name], weight)
                for name, weight in self.hyperparams["criteria_warped"]
            ]

            self.model.criteria_flow_nnf = [
                (CONFIGS_FLOW_LOSS[name], weight)
                for name, weight in self.hyperparams["criteria_flow"]
            ]

            self.model.registration_target_e = RegistrationTarget[self.hyperparams["registration_target"].upper()]
            self.model.registration_strategy_e = RegistrationStrategy[self.hyperparams["registration_strategy"].upper()]

            self.model.optimizer_nnf = CONFIGS_OPTIMIZER[self.hyperparams["optimizer"]]

        return self.model, self.hyperparams

    def set_network(self, config_identifier: str) -> TransMorphModuleBuilder:
        if not self.is_ckpt:
            print(f"network = {config_identifier}")
            self.hyperparams["network"] = config_identifier
        else:
            print("WARN: Cannot change network as it is a ckpt.")

        return self

    def set_criteria_warped(
        self, criteria_warped: str | list[tuple[str, float]]
    ) -> TransMorphModuleBuilder:
        if isinstance(criteria_warped, str):
            criteria = criteria_warped.split("-")
        else:
            criteria = [value for entry in criteria_warped for value in entry]

        print(f"criteria_warped = {criteria}")

        criteria_warped = [
            (criteria[i], float(criteria[i + 1]))
            for i in range(0, len(criteria), 2)
        ]

        self.hyperparams["criteria_warped"] = criteria_warped

        return self

    def set_criteria_flow(
        self, criteria_flow: str | list[tuple[str, float]]
    ) -> TransMorphModuleBuilder:
        if isinstance(criteria_flow, str):
            criteria = criteria_flow.split("-")
        else:
            criteria = [value for entry in criteria_flow for value in entry]

        print(f"criteria_flow = {criteria}")

        criteria_flow = [
            (criteria[i], float(criteria[i + 1]))
            for i in range(0, len(criteria), 2)
        ]

        self.hyperparams["criteria_flow"] = criteria_flow

        return self

    def set_registration_strategy(
        self, registration_strategy: str
    ) -> TransMorphModuleBuilder:
        print(f"registration_strategy = {registration_strategy}")
        self.hyperparams["registration_strategy"] = registration_strategy
        return self

    def set_registration_target(
        self, registration_target: str
    ) -> TransMorphModuleBuilder:
        print(f"registration_strategy = {registration_target}")
        self.hyperparams["registration_target"] = registration_target
        return self

    def set_registration_depth(
        self, registration_depth: int
    ) -> TransMorphModuleBuilder:
        if not self.is_ckpt:
            print(f"registration_depth = {registration_depth}")
            self.hyperparams["registration_depth"] = registration_depth
        else:
            print("WARN: Cannot change registration_depth as it is a ckpt. Indirectly affects network.")
        return self

    def set_registration_stride(
        self, registration_stride: int
    ) -> TransMorphModuleBuilder:
        print(f"registration_stride = {registration_stride}")
        self.hyperparams["registration_stride"] = registration_stride
        return self

    def set_identity_loss(self, identity_loss: bool) -> TransMorphModuleBuilder:
        print(f"identity_loss = {identity_loss}")
        self.hyperparams["identity_loss"] = identity_loss
        return self

    def set_optimizer(self, optimizer: str) -> TransMorphModuleBuilder:
        print(f"optimizer = {optimizer}")
        optimizer_name = optimizer
        optimizer = CONFIGS_OPTIMIZER[optimizer]
        self.hyperparams["optimizer"] = optimizer
        return self

    def set_learning_rate(self, learning_rate: float) -> TransMorphModuleBuilder:
        print(f"learning_rate = {learning_rate}")
        self.hyperparams["learning_rate"] = learning_rate
        return self
