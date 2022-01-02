import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

from models import BaseModel, AdvancedModel
from data import device


class Generator:
    def __init__(self, word_map: tuple, model_params: dict, train_params: dict, type_model='basic'):
        self.idx2word, self.word2idx = word_map
        self.model = BaseModel(**model_params) if type_model == 'basic' else AdvancedModel(**model_params)

        # train params
        self.num_epochs = train_params['epochs']
        self.lr = train_params['lr']
        self.lr_decay = train_params['lr_decay']
        self.step_lr_decay = train_params['step_lr_decay']
        self.sample_every = train_params['sample_every']

        # other
        self.checkpoints = dict()

        self.type_model = type_model

    def train(self, train_loader: DataLoader):
        save_to = f'cache/{self.type_model}/Generator_from_training_temp.pkl'
        print(f'Started training {self.type_model} model')

        self.model.train()
        self.model = self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_lr_decay, gamma=self.lr_decay)
        criterion = nn.CrossEntropyLoss().to(device)

        self.checkpoints['loss'] = []
        self.checkpoints['accuracy'] = []

        best_loss, best_accuracy = 100, 0
        try:
            for epoch in range(1, self.num_epochs + 1):
                mean_loss, accuracy = train_epoch(self, train_loader, criterion, optimizer, scheduler, epoch)
                self.checkpoints['loss'].append(mean_loss)
                self.checkpoints['accuracy'].append(accuracy)
                if mean_loss <= best_loss:
                    best_loss, best_accuracy = mean_loss, accuracy
                    torch.save(self.model.state_dict(), f'cache/{self.type_model}/weights.pkl')
                    torch.save(self.checkpoints, f'cache/{self.type_model}/stats.pkl')
                    pickle.dump(self, open(save_to, 'wb'))

        except KeyboardInterrupt as e:
            print(f'Training interrupted by user, best loss {best_loss: .5f}')
        self.model.load_state_dict(torch.load(f'cache/{self.type_model}/weights.pkl'))
        self.model = self.model.to(device)
        self.checkpoints = torch.load(f'cache/{self.type_model}/stats.pkl')

    @torch.no_grad()
    def generate_sample(self, start_tokens='<s>', max_length=800, T=1.) -> str:
        """
        :param start_tokens: start tokens
        :param max_length: max amount of tokens allowed to generate
        :param T: temperature of the softmax
        :return: will generate new tunes until end token is found or max length is reached, return text of abc notation
        """
        return self.model.sample((self.idx2word, self.word2idx), start_tokens, max_length, T)

    @torch.no_grad()
    def extend_music(self, start_tokens, T=1.5, every=3) -> str:
        return self.model.extend_new_music((self.idx2word, self.word2idx), start_tokens, every, T) \
            if self.type_model == 'advanced' else start_tokens


def train_epoch(generator: Generator, train_loader: DataLoader, criterion, optimizer, scheduler, epoch) -> tuple:
    total_examples, sum_correct, sum_loss = 0, 0, 0
    pbar = tqdm(train_loader)
    for i, (text_input, text_target) in enumerate(pbar):
        if i % generator.sample_every == 0:
            print('\n\n', generator.generate_sample(), '\n\n')
            generator.model.train()

        text_input, text_target = text_input.to(device), text_target[0].to(device)
        new_piece, _ = generator.model(text_input)
        loss = criterion(new_piece, text_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        mean_loss = sum_loss / (i + 1)
        sum_correct += (new_piece.argmax(dim=1) == text_target).sum().item()
        total_examples += text_target.shape[0]
        accuracy = sum_correct / total_examples

        pbar.set_description(f'{epoch=}\t{mean_loss=: .4f}\t{accuracy=: .5f}')

    scheduler.step()
    return sum_loss/len(train_loader), sum_correct / total_examples
