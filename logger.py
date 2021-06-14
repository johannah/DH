from comet_ml import Experiment

from torch.utils.tensorboard import SummaryWriter
import json
import os
import shutil
import torch
import torchvision
import numpy as np


COMET_LOG_FREQ = 10
SOURCE_CODES = [
    'dh_utils.py',
    'replay_buffer.py',
    'TD3.py',
    'train_rl.py',
    'train_bc.py',
    'utils.py'
]


class Logger(object):
    def __init__(self, log_dir, use_tb=True, use_comet=False, project_name="DH"):
        self._log_dir = log_dir
        if use_tb:
            tb_dir = os.path.join(log_dir, 'tb')
            if os.path.exists(tb_dir):
                shutil.rmtree(tb_dir)
            self._sw = SummaryWriter(tb_dir)
        else:
            self._sw = None

        if use_comet:
            self._comet = Experiment(project_name=project_name)
            exp_name = log_dir.rpartition('/')[-1]
            self._comet.set_name(exp_name)
            self._try_comet_log_code(SOURCE_CODES)

            try:
                self._comet.log_other('SLURM_JOB_ID', os.environ['SLURM_JOB_ID'])
            except KeyError:
                print("The job is not running on a SLURM cluster.")
        else:
            self._comet = None

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def _try_sw_log_image(self, key, image, step):
        if self._sw is not None:
            assert image.dim() == 3
            grid = torchvision.utils.make_grid(image.unsqueeze(1))
            self._sw.add_image(key, grid, step)

    def _try_comet_log(self, key, value, step):
        if self._comet is not None and step % COMET_LOG_FREQ == 0:
            self._comet.log_metric(key, value, step)

    def _try_comet_log_hyper_params(self, hyper_params):
        if self._comet is not None:
            self._comet.log_parameters(hyper_params)

    def _try_comet_log_image(self, key, image, step):
        if self._comet is not None and step % COMET_LOG_FREQ == 0:
            assert image.dim() == 3
            if image.size()[0] == 3:
                self._comet.log_image(image.detach().cpu(), name=key, step=step, image_channels='first')

    def _try_comet_log_figure(self, key, fig, step):
        if self._comet is not None and step % COMET_LOG_FREQ == 0:
            self._comet.log_figure(key, fig, step=step)

    def _try_comet_log_code(self, file_names):
        if self._comet is not None:
            for f in file_names:
                self._comet.log_code(file_name=f)

    def _try_comet_log_model(self, name, file_or_folder):
        if self._comet is not None:
            self._comet.log_model(name, file_or_folder)

    def log(self, key, value, step):
        if type(value) == torch.Tensor:
            value = value.item()
        if isinstance(value, dict):
            for k, v in value.items():
                self._try_sw_log(f'{key}/{k}', v, step)
                self._try_comet_log(f'{key}/{k}', v, step)
        else:
            self._try_sw_log(key, value, step)
            self._try_comet_log(key, value, step)

    def log_image(self, key, image, step):
        self._try_sw_log_image(key, image, step)
        self._try_comet_log_image(key, image, step)

    def log_figure(self, key, fig, step):
        self._try_comet_log_figure(key, fig, step)

    def log_hyper_params(self, hyper_params):
        self._try_comet_log_hyper_params(hyper_params)
        with open(os.path.join(self._log_dir, 'args.json'), 'w') as f:
            json.dump(hyper_params, f, sort_keys=True, indent=4)

    def log_model(self, name, file_or_folder):
        self._try_comet_log_model(name, file_or_folder)