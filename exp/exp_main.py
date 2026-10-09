import importlib
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, test_params_flop, visual

warnings.filterwarnings("ignore")


MODEL_NAMES = [
    "Autoformer",
    "Transformer",
    "Informer",
    "DLinear",
    "NLinear",
    "Linear",
    "PatchTST",
    "SegRNN",
    "CycleNet",
    "iTransformer",
    "TimeXer",
    "TQNet",
    "TQDLinear",
    "TQPatchTST",
    "TQiTransformer",
    "Ourmodel",
    "OLinear"
]


def _load_model_module(name):
    try:
        return importlib.import_module("models.{}".format(name))
    except ModuleNotFoundError as exc:
        missing_name = exc.name or ""
        if missing_name == "models.{}".format(name) or missing_name.startswith("models.{}.".format(name)):
            return None
        raise


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {}
        for name in MODEL_NAMES:
            module = _load_model_module(name)
            if module is not None:
                model_dict[name] = module

        if self.args.model not in model_dict:
            available = ", ".join(sorted(model_dict.keys()))
            raise ValueError("Unknown model '{}'. Available models: {}".format(self.args.model, available))

        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def _is_failure_model(self):
        return self.args.model == "Ourmodel"

    def _model_ref(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _unpack_batch(self, batch):
        if len(batch) == 5:
            return batch
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        return batch_x, batch_y, batch_x_mark, batch_y_mark, None

    def _move_batch(self, batch):
        batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle = self._unpack_batch(batch)
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        if batch_cycle is not None:
            batch_cycle = batch_cycle.int().to(self.device)
        return batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle

    def _decoder_input(self, batch_y):
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
        dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        return dec_inp

    def _forward_model(self, batch_x, batch_x_mark=None, dec_inp=None, batch_y_mark=None, batch_cycle=None):
        if self.args.model == "Ourmodel":
            return self.model(batch_x)

        if any(substr in self.args.model for substr in {"CycleNet", "TQ"}):
            return self.model(batch_x, batch_cycle)
        if any(substr in self.args.model for substr in {"Linear", "MLP", "SegRNN", "TST"}):
            return self.model(batch_x)
        if self.args.output_attention:
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    def _target_slice(self, batch_y):
        f_dim = -1 if self.args.features == "MS" else 0
        return batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

    def _prediction_slice(self, outputs):
        f_dim = -1 if self.args.features == "MS" else 0
        return outputs[:, -self.args.pred_len :, f_dim:]

    def _align_prediction(self, pred, target):
        if pred.size(-1) == target.size(-1):
            return pred
        if pred.size(-1) == 1:
            return pred.expand_as(target)
        if target.size(-1) == 1:
            return pred[:, :, -1:]
        raise ValueError("Prediction channels ({}) do not match target channels ({})".format(pred.size(-1), target.size(-1)))

    def _failure_targets(self, outputs, batch_y):
        target = self._target_slice(batch_y)
        base_forecast = self._prediction_slice(outputs["base_forecast"])
        base_forecast = self._align_prediction(base_forecast, target)
        per_horizon_error = torch.mean(torch.abs(base_forecast - target), dim=-1, keepdim=True)
        threshold = getattr(self.args, "failure_threshold", 0.5)
        labels = (per_horizon_error > threshold).float()
        return base_forecast, target, labels, per_horizon_error

    def _failure_loss(self, outputs, batch_y):
        base_forecast, target, labels, per_horizon_error = self._failure_targets(outputs, batch_y)
        probs = outputs["failure_prob"].clamp(1e-6, 1.0 - 1e-6)
        alpha = float(getattr(self.args, "failure_alpha", 0.75))
        alpha = min(max(alpha, 1e-4), 1.0 - 1e-4)

        if getattr(self.args, "failure_loss", "focal") == "focal":
            gamma = float(getattr(self.args, "failure_gamma", 2.0))
            pt = probs * labels + (1.0 - probs) * (1.0 - labels)
            alpha_t = alpha * labels + (1.0 - alpha) * (1.0 - labels)
            cls_loss = -(alpha_t * torch.pow(1.0 - pt, gamma) * torch.log(pt)).mean()
        else:
            cls_loss = -(
                alpha * labels * torch.log(probs)
                + (1.0 - alpha) * (1.0 - labels) * torch.log(1.0 - probs)
            ).mean()

        forecast_loss = torch.mean((base_forecast - target) ** 2)
        total_loss = cls_loss + float(getattr(self.args, "failure_forecast_weight", 0.1)) * forecast_loss
        stats = {
            "cls_loss": cls_loss.detach(),
            "forecast_loss": forecast_loss.detach(),
            "failure_rate": labels.mean().detach(),
            "mean_error": per_horizon_error.mean().detach(),
        }
        return total_loss, stats

    def _train_profile_from_data(self, data_set):
        data = getattr(data_set, "data_x", None)
        if data is None:
            return np.zeros(8, dtype=np.float32)

        x = np.asarray(data, dtype=np.float64)
        eps = 1e-6
        mean = float(np.mean(x))
        std = float(np.std(x) + eps)
        normed = (x - mean) / std
        skew = float(np.mean(normed ** 3))
        kurt = float(np.mean(normed ** 4))

        if x.shape[0] > 1:
            x0 = x[:-1] - np.mean(x[:-1], axis=0, keepdims=True)
            x1 = x[1:] - np.mean(x[1:], axis=0, keepdims=True)
            acf = np.mean((x0 * x1).mean(axis=0) / (x0.std(axis=0) * x1.std(axis=0) + eps))
            t = np.linspace(-1.0, 1.0, x.shape[0]).reshape(-1, 1)
            slope = np.sum((x - x.mean(axis=0, keepdims=True)) * t, axis=0) / (np.sum(t * t) + eps)
            slope_abs = float(np.mean(np.abs(slope)))
        else:
            acf = 0.0
            slope_abs = 0.0

        residual = x - np.mean(x, axis=0, keepdims=True)
        seasonality = float(np.std(residual) / std)
        residual_energy = float(np.mean(residual ** 2))
        return np.array([mean, std, skew, kurt, acf, slope_abs, seasonality, residual_energy], dtype=np.float32)

    def _maybe_set_train_profile(self, train_data):
        if not self._is_failure_model():
            return
        model = self._model_ref()
        if hasattr(model, "set_train_profile"):
            model.set_train_profile(self._train_profile_from_data(train_data))

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch in vali_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle = self._move_batch(batch)
                dec_inp = self._decoder_input(batch_y)

                outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
                if self._is_failure_model():
                    loss, _ = self._failure_loss(outputs, batch_y)
                else:
                    outputs = self._prediction_slice(outputs)
                    batch_y = self._target_slice(batch_y)
                    loss = criterion(outputs.detach().cpu(), batch_y.detach().cpu())
                total_loss.append(loss.item() if torch.is_tensor(loss) else loss)

        self.model.train()
        return np.average(total_loss)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")
        self._maybe_set_train_profile(train_data)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle = self._move_batch(batch)
                dec_inp = self._decoder_input(batch_y)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
                        if self._is_failure_model():
                            loss, _ = self._failure_loss(outputs, batch_y)
                        else:
                            outputs = self._prediction_slice(outputs)
                            batch_y = self._target_slice(batch_y)
                            loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
                    if self._is_failure_model():
                        loss, _ = self._failure_loss(outputs, batch_y)
                    else:
                        outputs = self._prediction_slice(outputs)
                        batch_y = self._target_slice(batch_y)
                        loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                    loss.backward()
                    model_optim.step()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.lradj == "TST":
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != "TST":
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print("Updating learning rate to {}".format(scheduler.get_last_lr()[0]))

        best_model_path = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def _failure_metrics(self, probs, labels):
        probs = probs.reshape(-1)
        labels = labels.reshape(-1).astype(np.int64)
        pred = (probs >= getattr(self.args, "failure_decision_threshold", 0.5)).astype(np.int64)

        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        tn = np.sum((pred == 0) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))
        eps = 1e-12
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        acc = (tp + tn) / (tp + tn + fp + fn + eps)
        brier = np.mean((probs - labels) ** 2)
        return {
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "brier": brier,
            "failure_rate": float(np.mean(labels)),
            "pred_failure_rate": float(np.mean(pred)),
        }

    def _test_failure_model(self, setting, test_loader):
        probs_list, labels_list, base_list, true_list, error_list = [], [], [], [], []
        folder_path = "./test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle = self._move_batch(batch)
                dec_inp = self._decoder_input(batch_y)
                outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
                base_forecast, target, labels, errors = self._failure_targets(outputs, batch_y)

                probs = outputs["failure_prob"].detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                base = base_forecast.detach().cpu().numpy()
                true = target.detach().cpu().numpy()
                errors = errors.detach().cpu().numpy()

                probs_list.append(probs)
                labels_list.append(labels)
                base_list.append(base)
                true_list.append(true)
                error_list.append(errors)

                if i % 20 == 0:
                    input_x = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input_x[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input_x[0, :, -1], base[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

        probs = np.concatenate(probs_list, axis=0).reshape(-1, self.args.pred_len, 1)
        labels = np.concatenate(labels_list, axis=0).reshape(-1, self.args.pred_len, 1)
        base_preds = np.concatenate(base_list, axis=0)
        trues = np.concatenate(true_list, axis=0)
        errors = np.concatenate(error_list, axis=0).reshape(-1, self.args.pred_len, 1)

        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + "failure_prob.npy", probs)
        np.save(folder_path + "failure_label.npy", labels)
        np.save(folder_path + "failure_error.npy", errors)
        np.save(folder_path + "base_pred.npy", base_preds)
        np.save(folder_path + "true.npy", trues)

        mae, mse, rmse, mape, mspe, rse, corr = metric(base_preds, trues)
        cls = self._failure_metrics(probs, labels)
        print(
            "base_mse:{}, base_mae:{} | fail_acc:{:.4f}, precision:{:.4f}, recall:{:.4f}, f1:{:.4f}, brier:{:.4f}".format(
                mse, mae, cls["acc"], cls["precision"], cls["recall"], cls["f1"], cls["brier"]
            )
        )

        with open("result.txt", "a") as f:
            f.write(setting + "  \n")
            f.write(
                "base_mse:{}, base_mae:{} | fail_acc:{:.4f}, precision:{:.4f}, recall:{:.4f}, f1:{:.4f}, brier:{:.4f}, failure_rate:{:.4f}".format(
                    mse, mae, cls["acc"], cls["precision"], cls["recall"], cls["f1"], cls["brier"], cls["failure_rate"]
                )
            )
            f.write("\n\n")

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")

        if test:
            print("loading model")
            checkpoint_path = os.path.join("./checkpoints/" + setting, "checkpoint.pth")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        if self._is_failure_model():
            self._test_failure_model(setting, test_loader)
            return

        preds, trues = [], []
        folder_path = "./test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle = self._move_batch(batch)
                dec_inp = self._decoder_input(batch_y)
                outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)

                outputs = self._prediction_slice(outputs)
                batch_y = self._target_slice(batch_y)
                outputs_np = outputs.detach().cpu().numpy()
                batch_y_np = batch_y.detach().cpu().numpy()

                preds.append(outputs_np)
                trues.append(batch_y_np)
                if i % 20 == 0:
                    input_x = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input_x[0, :, -1], batch_y_np[0, :, -1]), axis=0)
                    pd = np.concatenate((input_x[0, :, -1], outputs_np[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()

        preds = np.concatenate(preds, axis=0).reshape(-1, preds[0].shape[-2], preds[0].shape[-1])
        trues = np.concatenate(trues, axis=0).reshape(-1, trues[0].shape[-2], trues[0].shape[-1])

        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print("mse:{}, mae:{}".format(mse, mae))
        with open("result.txt", "a") as f:
            f.write(setting + "  \n")
            f.write("mse:{}, mae:{}".format(mse, mae))
            f.write("\n\n")

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag="pred")

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = os.path.join(path, "checkpoint.pth")
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        preds = []
        probs = []
        self.model.eval()
        with torch.no_grad():
            for batch in pred_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle = self._move_batch(batch)
                dec_inp = self._decoder_input(batch_y)
                outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
                if self._is_failure_model():
                    probs.append(outputs["failure_prob"].detach().cpu().numpy())
                    preds.append(outputs["base_forecast"].detach().cpu().numpy())
                else:
                    preds.append(outputs.detach().cpu().numpy())

        preds = np.array(preds).reshape(-1, preds[0].shape[-2], preds[0].shape[-1])
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + "real_prediction.npy", preds)
        if probs:
            probs = np.array(probs).reshape(-1, probs[0].shape[-2], probs[0].shape[-1])
            np.save(folder_path + "real_failure_prob.npy", probs)
