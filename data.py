import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, AnyStr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyDataset(Dataset):
    def __init__(self, folder_path: str):
        self.dataset, self.word_map = self.data_prep(folder_path)

    @staticmethod
    def data_prep_text(file_path):
        tunes = []
        unique_words = set()
        with open(file_path, 'r') as file:
            tune_with_details = []
            for i, line in enumerate(file, start=1):
                splitted = line.split(" ") if i % 4 != 3 else line[:-2].split(" ")
                if i % 4 == 0:
                    tunes.append(tune_with_details)
                    tune_with_details = []
                else:
                    unique_words = unique_words.union(set(splitted))
                    tune_with_details += splitted
        return tunes, unique_words

    @staticmethod
    def data_prep(file_path) -> Tuple[List[List], Tuple[List[AnyStr], Dict]]:
        tunes, unique_words = MyDataset.data_prep_text(file_path)
        idx2word = ["<s>", "</s>"] + list(unique_words)
        word2idx = {word: i for i, word in enumerate(idx2word)}
        word_map = (idx2word, word2idx)
        tunes = [[word2idx["<s>"]] + [word2idx[word] for word in tune] + [word2idx["</s>"]] for tune in tunes]
        return tunes, word_map

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.dataset[index])
        return x[:-1].long(), x[1:].long()

    def __len__(self) -> int:
        return len(self.dataset)

