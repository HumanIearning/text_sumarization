import numpy as np

import torch
from torch import optim
import torch.nn.utils as torch_utils

from ignite.engine import Engine
from ignite.engine import Events

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

class MyEngine(Engine):
    def __init__(self, func, model, crit, optimizer, config):
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.config = config

        super(MyEngine, self).__init__(func)

        self.best_loss = np.inf
        self.scaler = GradScaler()

    @staticmethod
    def train(engine, mini_batch):
        engine.model.train()

        engine.optimizer.zero_grad()

        device = next(engine.model.parameters()).device
        # device = torch.device("mps")

        x = (mini_batch['src'].to(device), mini_batch['src_len'])
        y = mini_batch['tgt'].to(device)

        print(y)
        with autocast():
            y_hat = engine.model(x, y)
            # |y_hat| = (bs, len, output_size)  Probability distribution

            print(y_hat)

            for i in engine.model.named_parameters():
                print(i)
                break


            loss = engine.crit(
                y_hat.contiguous().view(-1, y_hat[0].size(-1)),
                y.contiguous().view(-1)
            )

            loss_mean = loss.div(y[0].size(0))

        # engine.scaler.scale(loss_mean).backward()
        loss_mean.backward()

        word_count = int(sum(x[1]))

        # torch_utils.clip_grad_norm(
        #     engine.model.parameters(),
        #     engine.config.max_grad_norm
        # )

        # engine.scaler.step(engine.optimizer)
        # engine.scaler.update()
        engine.optimizer.step()

        loss = float(loss / word_count)
        ppl = np.exp(loss)

        return {
            'loss' : loss,
            'ppl' : ppl
        }


    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            device = next(engine.model.parameters()).device

            x = (mini_batch['src'].to(device), mini_batch['src_len'])
            y = (mini_batch['tgt'].to(device), mini_batch['tgt_len'])

            y_hat = engine.model(x, y[0])

        loss = engine.crit(
            y_hat.contiguous().view(-1, y_hat[0].size(-1)),
            y.contiguous().view(-1)
        )

        ppl = np.exp(loss)

        return {
            'loss': loss,
            'ppl': ppl
        }



    @staticmethod
    def attach(train_engine, valid_engine):
        train_metric_names = ['loss', 'ppl'] #, '|param|', '|g_param|']
        valid_metric_names = ['loss', 'ppl']

        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name
            )

        for name in train_metric_names:
            attach_running_average(train_engine, name)

        pbar = ProgressBar(bar_format=None, ncols=120)
        pbar.attach(train_engine, train_metric_names)

        @train_engine.on(Events.EPOCH_COMPLETED)
        def print_train_logs(engine):
            # avg_p_norm = engine.state.metrics['|param|']
            # avg_g_norm = engine.state.metrics['|g_param|']
            avg_loss = engine.state.metrics['loss']
            avg_ppl = engine.state.metrics['ppl']

            print('Epoch {} - '
                  # '|param|={:.2e} |g_param|={:.2e} '
                  'loss={:.4e} ppl={:.2f}'.format(
                engine.state.epoch,
                # avg_p_norm,
                # avg_g_norm,
                avg_loss,
                avg_ppl,
            ))

        for name in valid_metric_names:
            attach_running_average(valid_engine, name)

        pbar = ProgressBar(bar_format=None, ncols=120)
        pbar.attach(valid_engine, valid_metric_names)

        @valid_engine.on(Events.EPOCH_COMPLETED)
        def print_valid_logs(engine):
            avg_loss = engine.state.metrics['loss']
            avg_ppl = engine.state.metrics['ppl']

            print('Epoch {} - loss={:.4e} ppl={:.2f} best_loss={:.4e} best_ppl={:.2f}'.format(
                avg_loss,
                avg_ppl,
                engine.best_loss,
                np.exp(engine.best_loss),
            ))
    @staticmethod
    def check_best(engine):
        if engine.state.metrics['loss'] < engine.best_loss:
            engine.best_loss = engine.state.metrics['loss']

    @staticmethod
    def save_model(engine, train_engine, config, src_vocab, tgt_vocab):
        train_loss = train_engine.state.metrics['loss']
        valid_loss = engine.state.metrics['loss']

        model_fn = config.model_fn.split('.')
        model_fn = model_fn +  ['epoch:%02d-' % train_engine.state.epoch, 'loss:train-%.2f.valid-%.2f' % (train_loss, valid_loss)]
        model_fn = '.'.join(model_fn)

        torch.save(
            {
                'model' : engine.model.state_dict(),
                'optim' : train_engine.optimizer.state_dict(),
                'src_vocab' : src_vocab,
                'tgt_vocab' : tgt_vocab
            }, model_fn
        )

class Trainer():
    def __init__(self, engine, config):
        self.engine = engine
        self.config = config

    def train(self, model, crit, optimizer,
              train_loader, valid_loader,
              src_vocab, tgt_vocab,
              n_epochs):
        train_engine = self.engine(
            self.engine.train,
            model,
            crit,
            optimizer,
            self.config
        )
        valid_engine = self.engine(
            self.engine.validate,
            model,
            crit,
            optimizer=None,
            config=self.config
        )
        self.engine.attach(
            train_engine,
            valid_engine
        )

        def run_validation(valid_engine, valid_loader):
            valid_engine.run(valid_loader, max_epochs=1)

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            run_validation,
            valid_engine,
            valid_loader
        )

        valid_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            self.engine.check_best
        )

        valid_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            self.engine.save_model,
            train_engine,
            self.config,
            src_vocab,
            tgt_vocab
        )

        train_engine.run(train_loader, max_epochs=n_epochs)

