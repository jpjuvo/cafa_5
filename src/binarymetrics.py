import torchmetrics
import torch
import numpy as np

class BinaryMetrics:

    def __init__(self, device='cuda'):
        self.metrics = [
            ('auc', torchmetrics.AUROC(task='binary').to(device)),
            ('f1', torchmetrics.F1Score(task='binary', threshold=0.5).to(device)),
            ('stats', torchmetrics.StatScores(task='binary', threshold=0.5).to(device))
        ]

    def accumulate_batch(self, preds, targets):
        """ Update metrics with a batch """
        for _, metric in self.metrics:
            # make sure everything is binary
            _ = metric(preds.contiguous().view(-1, 1), targets.contiguous().view(-1, 1))

    def compute(self, logs, prefix='train_'):
        """ log metrics and reset accumulated batches """
        for metric_name, metric in self.metrics:
            if metric_name == 'stats':
                # extract Precision, Recall and Acc
                [tp, fp, tn, fn, sup] = metric.compute().cpu().numpy()
                logs[f'{prefix}precision'] = tp / (tp + fp)
                logs[f'{prefix}recall'] = tp / (tp + fn)
                logs[f'{prefix}accuracy'] = (tp+tn) / (tp + tn + fp + fn)
            else:
                logs[f'{prefix}{metric_name}'] = float(metric.compute().cpu().numpy())

        self._reset_metrics()

    def _reset_metrics(self):
        """ Reset accumulated stats """
        for _, metric in self.metrics:
            metric.reset()
