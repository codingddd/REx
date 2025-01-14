from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from rex import accelerator
from rex.data.data_manager import DataManager
from rex.data.transforms.base import TransformBase
from rex.metrics import safe_division
from rex.tasks.base_task import TaskBase
from rex.utils.config import DefaultBaseConfig
from rex.utils.dict import get_dict_content
from rex.utils.io import dump_json
from rex.utils.logging import logger
from rex.utils.param import calc_module_params
from rex.utils.progress_bar import pbar
from rex.utils.wrapper import safe_try


class SimpleTask(TaskBase):
    """Simple task with one model, one optimizer and (maybe) one lr scheduler"""

    def __init__(
        self,
        config: Union[DefaultBaseConfig, DictConfig],
        initialize: Optional[bool] = True,
        makedirs: Optional[bool] = True,
        dump_configfile: Optional[bool] = True,
    ) -> None:
        super().__init__(config, initialize, makedirs, dump_configfile)

        self.middle_path = Path(config.task_dir).joinpath("middle")
        self.middle_path.mkdir(parents=True, exist_ok=True)
        self.measures_path = Path(config.task_dir).joinpath("measures")
        self.measures_path.mkdir(parents=True, exist_ok=True)

        self.optimizer = None
        self.lr_scheduler = None
        self.transform: TransformBase
        self.data_manager: DataManager
        self.model: nn.Module

        self.initialize()
        self.after_initialization()

    def initialize(self):
        logger.debug("Init transform")
        self.transform: TransformBase = self.init_transform()
        logger.debug(f"transform: {type(self.transform)}")
        logger.debug("Init data_manager")
        self.data_manager: DataManager = self.init_data_manager()
        logger.debug(f"data manager: {type(self.data_manager)}")

        logger.debug("Init model")
        self.model = self.init_model()
        logger.debug(f"model: {type(self.model)}")
        logger.debug(f"model: {self.model}")
        num_model_params = calc_module_params(self.model)
        logger.debug(f"#ModelParams: {num_model_params}")
        logger.debug("Prepare model")
        self.model = accelerator.prepare_model(self.model)

    def after_initialization(self):
        pass

    def init_transform(self) -> TransformBase:
        raise NotImplementedError

    def init_data_manager(self) -> DataManager:
        raise NotImplementedError

    def init_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def init_optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    def init_lr_scheduler(self) -> Union[None, torch.optim.lr_scheduler._LRScheduler]:
        return None

    def after_whole_train(self):
        pass

    @torch.no_grad()
    def _get_eval_results_impl(
        self, input_batches: List, output_batches: List, *args, **kwargs
    ) -> dict:
        return self.get_eval_results(input_batches, output_batches, *args, **kwargs)

    def get_eval_results(
        self, input_batches: List, output_batches: List, *args, **kwargs
    ) -> dict:
        """Get evaluation measurements

        Args:
            input_batches: list of model input. Raw batch input.
            output_batches: list of model results (`preds`)

        Returns:
            ``rex.utils.dict.PrettyPrintDefaultDict``
                and ``rex.utils.dict.PrettyPrintDict`` is highly recommended
                to replace vanilla ``defaultdict`` and ``dict`` here.
        """
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, *args, **kwargs):
        self.model.eval()
        return self.predict_api(*args, **kwargs)

    def predict_api(self, *args, **kwargs):
        raise NotImplementedError

    def get_data_loader(self, dataset_name, is_eval="guessing", epoch=0):
        loader = self.data_manager.load_loader(
            dataset_name, is_eval=is_eval, epoch=epoch
        )
        return loader

    def log_loss(
        self, idx: int, loss_item: float, step_or_epoch: str, dataset_name: str
    ):
        pass

    def log_metrics(
        self, idx: int, metrics: dict, step_or_epoch: str, dataset_name: str
    ):
        pass

    # @safe_try
    def train(self):
        if self.config.skip_train:
            raise RuntimeError(
                "Training procedure started while config.skip_train is True!"
            )
        else:
            logger.debug("Init optimizer")
            self.optimizer = self.init_optimizer()
            logger.debug(f"optimizer: {self.optimizer}")
            logger.debug("Prepare optimizer")
            self.optimizer = accelerator.prepare_optimizer(self.optimizer)
            logger.debug("Init lr_scheduler")
            self.lr_scheduler = self.init_lr_scheduler()
            if self.lr_scheduler is not None:
                logger.debug(f"lr_scheduler: {type(self.lr_scheduler)}")
                logger.debug("Prepare lr_scheduler")
                self.lr_scheduler = accelerator.prepare_scheduler(self.lr_scheduler)

        if self.config.resumed_training_path is not None:
            self.load(
                self.config.resumed_training_path,
                load_config=False,
                load_model=True,
                load_optimizer=True,
                load_history=True,
            )
            resumed_training = True
        else:
            resumed_training = False
        train_loader = self.get_data_loader("train", False, self.history["curr_epoch"])
        total_steps = self.history["curr_epoch"] * len(train_loader)
        start_time = datetime.now()
        for epoch_idx in range(self.history["curr_epoch"], self.config.num_epochs):
            logger.info(f"Start training {epoch_idx}/{self.config.num_epochs}")
            if not resumed_training:
                self.history["curr_epoch"] = epoch_idx

            used_time = datetime.now() - start_time
            time_per_epoch = safe_division(used_time, self.history["curr_epoch"])
            remain_time = time_per_epoch * (
                self.config.num_epochs - self.history["curr_epoch"]
            )
            logger.info(
                f"Epoch: {epoch_idx}/{self.config.num_epochs} [{str(used_time)}<{str(remain_time)}, {str(time_per_epoch)}/epoch]"
            )

            self.model.train()
            self.optimizer.zero_grad()
            epoch_train_loader = self.get_data_loader(
                "train", is_eval=False, epoch=epoch_idx
            )
            loader = pbar(epoch_train_loader, desc=f"Train(e{epoch_idx})")
            for batch_idx, batch in enumerate(loader):
                if not resumed_training:
                    self.history["curr_batch"] = batch_idx
                    self.history["total_steps"] = total_steps
                if resumed_training and total_steps < self.history["total_steps"]:
                    total_steps += 1
                    continue
                elif resumed_training and total_steps == self.history["total_steps"]:
                    resumed_training = False

                result = self.model(**batch)

                result["loss"] /= self.config.grad_accum_steps
                accelerator.backward(result["loss"])
                loss_item = result["loss"].item()
                self.history["current_train_loss"]["epoch"] += loss_item
                self.history["current_train_loss"]["step"] += loss_item
                loader.set_postfix({"loss": loss_item})
                self.log_loss(self.history["total_steps"], loss_item, "step", "train")

                if self.config.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.config.max_grad_norm
                    )
                if ((batch_idx + 1) % self.config.grad_accum_steps) == 0 or (
                    batch_idx + 1
                ) == len(loader):
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                if (
                    self.config.step_eval_interval > 0
                    and (self.history["total_steps"] + 1)
                    % self.config.step_eval_interval
                    == 0
                ):
                    self._eval_during_train("step")
                    if not self._check_patience():
                        break
                total_steps += 1
                if not resumed_training:
                    self.history["total_steps"] += 1

            logger.info(loader)
            if (self.config.epoch_eval_interval > 0) and (
                ((epoch_idx + 1) % self.config.epoch_eval_interval) == 0
            ):
                self._eval_during_train("epoch")
                if not self._check_patience():
                    break

        logger.info("Trial finished.")
        if self.config.select_best_by_key == "metric":
            tmp_string = f"{self.history['best_metric']:.5f}"
        else:
            tmp_string = f"{self.history['best_loss']}"
        logger.info(
            f"Best epoch: {self.history['best_epoch']}, step: {self.history['best_step']}"
        )
        logger.info(
            f"Best {self.config.select_best_on_data}.{self.config.select_best_by_key}.{self.config.best_metric_field} : {tmp_string}"
        )

        if self.config.final_eval_on_test:
            logger.info("Loading best ckpt")
            self.load_best_ckpt()
            test_loss, test_measures = self.eval(
                "test", verbose=True, dump=True, postfix="final"
            )
            self.log_loss(0, test_loss, "final", "test")
            self.log_metrics(0, test_measures, "final", "test")
            return test_loss, test_measures

        self.after_whole_train()

        return self.history["best_loss"], self.history["best_metric"]

    def _eval_during_train(self, eval_on: Optional[str] = "epoch"):
        """Evaluation during training to record eval info, and control training process

        Args:
            eval_on: epoch or step
        """
        curr_epoch_idx = self.history["curr_epoch"]
        curr_total_steps = self.history["total_steps"]
        history_idx = curr_epoch_idx if eval_on == "epoch" else curr_total_steps
        history_idx_identifier = f"{eval_on}.{history_idx}"
        logger.info(f"Start evaluating at {history_idx_identifier}")

        # make sure the model save at least one checkpoint no matter
        #   evaluation or not
        if self.config.save_every_ckpt:
            self.save_ckpt(f"{history_idx_identifier}")

        # validate
        if eval_on not in ["epoch", "step"]:
            raise ValueError(f"eval_on: {eval_on} must eq `epoch` or `step`")
        if self.config.select_best_by_key not in ["metric", "loss"]:
            raise ValueError(
                f"select_best_by_key is {self.config.select_best_by_key}, while candidates are: `metric` and `loss`"
            )
        if self.config.select_best_on_data not in self.config.eval_on_data:
            raise ValueError(
                f"{self.config.select_best_on_data} is not included in eval_on_data: {self.config.eval_on_data}"
            )
        if len(self.config.eval_on_data) < 1:
            logger.warning(
                "Does not provide any data to evaluate during training",
            )

        # init
        this_eval_result = {}  # to dump results

        # eval to get measurements and loss
        eval_on_datasets = set()
        for dataset_name in self.config.eval_on_data:
            dataset_name = DataManager._get_normalized_dataset_name(dataset_name)
            if dataset_name != "train":
                eval_on_datasets.add(dataset_name)
            elif self.config.select_best_by_key != "loss":
                raise ValueError(
                    "Only `select_best_by_key=loss` is allowed when `eval_on_data=train`"
                )
        for dataset_name in eval_on_datasets:
            eval_loss, eval_measures = self.eval(
                dataset_name, verbose=False, postfix=history_idx_identifier
            )
            self.log_loss(history_idx, eval_loss, eval_on, dataset_name)
            self.log_metrics(history_idx, eval_measures, eval_on, dataset_name)
            self.history[eval_on][dataset_name]["metrics"][history_idx] = eval_measures
            self.history[eval_on][dataset_name]["loss"][history_idx] = eval_loss
            this_eval_result[f"{eval_on}.{dataset_name}.metrics"] = eval_measures
            this_eval_result[f"{eval_on}.{dataset_name}.loss"] = eval_loss

        # update the best
        select_best_on_data = DataManager._get_normalized_dataset_name(
            self.config.select_best_on_data
        )
        metric = -1.0
        is_best_metric = False
        if not (
            select_best_on_data == "train" and self.config.select_best_by_key == "loss"
        ):
            metric = self.history[eval_on][select_best_on_data]["metrics"][history_idx]
            metric = get_dict_content(metric, self.config.best_metric_field)
            if metric > self.history["best_metric"]:
                is_best_metric = True
                self.history["best_metric"] = metric
            this_eval_result["is_best_metric"] = is_best_metric

        this_eval_result["train_loss"] = self.history["current_train_loss"][eval_on]
        self.log_loss(history_idx, this_eval_result["train_loss"], eval_on, "sum_train_loss")
        if select_best_on_data == "train" and self.config.select_best_by_key == "loss":
            loss = this_eval_result["train_loss"]
        else:
            loss = self.history[eval_on][select_best_on_data]["loss"][history_idx]
        is_best_loss = False
        if loss < self.history["best_loss"]:
            is_best_loss = True
            self.history["best_loss"] = loss
        this_eval_result["is_best_loss"] = is_best_loss

        # count no climbing
        is_best = False
        if (self.config.select_best_by_key == "metric" and is_best_metric) or (
            self.config.select_best_by_key == "loss" and is_best_loss
        ):
            is_best = True
            self.history["best_epoch"] = curr_epoch_idx
            self.history["best_step"] = curr_total_steps
            if eval_on == "epoch":
                self.history["no_climbing_epoch_cnt"] = 0
            else:
                self.history["no_climbing_step_cnt"] = 0
        else:
            if eval_on == "epoch":
                self.history["no_climbing_epoch_cnt"] += self.config.epoch_eval_interval
            else:
                self.history["no_climbing_step_cnt"] += self.config.step_eval_interval
        this_eval_result["is_best"] = is_best
        this_eval_result["no_climbing_epoch_cnt"] = self.history[
            "no_climbing_epoch_cnt"
        ]
        this_eval_result["no_climbing_step_cnt"] = self.history["no_climbing_step_cnt"]

        # print results
        logger.info(
            f"Eval on {eval_on}, Idx: {history_idx_identifier}, is_best: {is_best}"
        )
        logger.info(f"Train loss: {this_eval_result['train_loss']:.5f}")
        for dataset_name in eval_on_datasets:
            eval_measures = self.history[eval_on][dataset_name]["metrics"][history_idx]
            eval_loss = self.history[eval_on][dataset_name]["loss"][history_idx]
            logger.info(
                (
                    f"{dataset_name} - "
                    f"{self.config.best_metric_field}: "
                    f"{get_dict_content(eval_measures, self.config.best_metric_field):.5f}, "
                    f"eval loss: {eval_loss:.5f}"
                )
            )
        if self.config.select_best_by_key == "metric":
            tmp_string = f"{self.history['best_metric']:.5f}"
        else:
            tmp_string = f"{self.history['best_loss']:.5f}"
        logger.info(f"Best {self.config.select_best_by_key}: {tmp_string}")
        logger.info(f"Best {eval_on}: {self.history[f'best_{eval_on}']}")
        logger.info(
            f"No Climbing Count of {eval_on}: {self.history[f'no_climbing_{eval_on}_cnt']}"
        )
        dump_json(
            this_eval_result,
            self.measures_path.joinpath(f"{history_idx_identifier}.json"),
            indent=2,
        )

        # reset current training loss
        self.history["current_train_loss"][eval_on] = 0.0

        # save checkpoints
        if is_best:
            if self.config.save_best_ckpt:
                self.save_ckpt("best")
            if self.config.save_best_ckpt == "all":
                self.save_ckpt(f"best.{history_idx_identifier}")

    def _check_patience(self):
        """Check patience, returns False if training process should be stopped, else returns True"""
        if (
            self.config.epoch_patience > 0
            and self.history["no_climbing_epoch_cnt"] >= self.config.epoch_patience
        ) or (
            self.config.step_patience > 0
            and self.history["no_climbing_step_cnt"] >= self.config.step_patience
        ):
            logger.info(
                (
                    "Early Stopped: No climbing count: "
                    f"Epoch: {self.history['no_climbing_epoch_cnt']} / {self.config.epoch_patience} "
                    f"Step: {self.history['no_climbing_step_cnt']} / {self.config.step_patience} "
                )
            )
            return False
        if (
            self.config.num_steps > 0
            and self.history["total_steps"] >= self.config.num_steps
        ):
            logger.info(
                f"Reached the max num of steps: {self.history['total_steps']} / {self.config.num_steps}"
            )
            return False
        return True

    @torch.no_grad()
    def eval(
        self, dataset_name, verbose=False, dump=False, postfix=""
    ) -> Tuple[float, dict]:
        """Eval on specific dataset and return loss and measurements

        Args:
            dataset_name: which dataset to evaluate
            verbose: whether to log evaluation results
            dump: if True, dump result to this filepath
            postfix: filepath postfix for dumping

        Returns:
            eval_loss: float
            metrics: dict
        """
        self.model.eval()
        eval_loader = self.get_data_loader(
            dataset_name, is_eval=True, epoch=self.history["curr_epoch"]
        )
        loader = pbar(eval_loader, desc=f"{dataset_name} - {postfix} Eval", ascii=True)

        eval_loss = 0.0
        metrics = {}
        origin = []
        output = []
        # raw_batch: dict
        for batch in loader:
            out = self.model(**batch, is_eval=True)
            eval_loss += out["loss"].item()
            origin.append(batch)
            output.append(out)

        logger.info(loader)
        metrics = self._get_eval_results_impl(origin, output, postfix)

        if verbose:
            logger.info(f"Eval dataset: {dataset_name}")
            logger.info(f"Eval loss: {eval_loss}")
            logger.info(
                f"Eval metrics: {get_dict_content(metrics, self.config.best_metric_field)}"
            )
        if dump:
            dump_obj = {
                "dataset_name": dataset_name,
                "eval_loss": eval_loss,
                "metrics": metrics,
            }
            dump_json(
                dump_obj, self.measures_path.joinpath(f"{dataset_name}.{postfix}.json")
            )

        return eval_loss, metrics
