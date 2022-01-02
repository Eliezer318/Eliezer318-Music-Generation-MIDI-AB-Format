from torch.utils.data import DataLoader
import pickle
from music21 import converter
import os
from matplotlib import pyplot as plt

from data import MyDataset
from train import Generator


def plot_graphs(accuracy: list, loss: list, model_type='Basic'):
    plt.plot(accuracy[:20])
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title(f'Accuracy Over Epochs - {model_type} Model')
    plt.plot()
    plt.show()

    plt.figure()

    plt.plot(loss[:20])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(f'Loss Over Epochs - {model_type} Model')
    plt.show()


def plot_both():
    gen = pickle.load(open('cache/basic/final_generator.pkl', 'rb'))
    ad_gen = pickle.load(open('cache/advanced/final_generator.pkl', 'rb'))
    plt.plot(gen.checkpoints['accuracy'][:20], label='Basic Model')
    plt.plot(ad_gen.checkpoints['accuracy'][:20], label='Advanced Model')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title(f'Accuracy Over Epochs')
    plt.legend()
    plt.show()

    plt.figure()

    plt.plot(gen.checkpoints['loss'][:20], label='Basic Model')
    plt.plot(ad_gen.checkpoints['loss'][:20], label='Advanced Model')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(f'Loss Over Epochs')
    plt.legend()
    plt.show()


def convert_text2midi(generator: Generator, save_to: str, amount=20):
    count = 0
    for i in range(1, amount + 1):
        try:
            text = generator.generate_sample()
            s = converter.parse(text)
            s.write('midi', fp=f'{save_to}/output{i}.midi')
        except:
            count += 1
    print(f'Encounter {count} problematic music texts')


def basic_part(dataset: MyDataset, data_loader: DataLoader, epochs=500, folder_path='data_v2.txt'):
    os.makedirs('samples/basic', exist_ok=True)
    os.makedirs('cache/basic', exist_ok=True)
    saved_to = 'cache/basic/final_generator.pkl'

    vocab_size = len(dataset.word_map[0])

    train_params = dict(epochs=epochs, lr=0.005, lr_decay=0.1, step_lr_decay=8, sample_every=3500)
    model_params = dict(vocab_size=vocab_size, input_size=vocab_size, hidden_size=262, num_layers=3, dropout=0.1)

    # gen = Generator(word_map=dataset.word_map, model_params=model_params, train_params=train_params)
    gen: Generator = pickle.load(open(saved_to, 'rb'))
    # gen.train(data_loader)
    plot_graphs(gen.checkpoints['accuracy'], gen.checkpoints['loss'], 'Basic')

    # pickle.dump(gen, open(save_to, 'wb'))
    # convert_text2midi(gen, 'samples/basic')


def advanced_part(dataset: MyDataset, data_loader: DataLoader, epochs=500):
    os.makedirs('samples/advanced', exist_ok=True)
    os.makedirs('cache/advanced', exist_ok=True)

    saved_to = f'cache/advanced/final_generator.pkl'

    train_params = dict(epochs=epochs, lr=8e-5, lr_decay=0.95, step_lr_decay=1, sample_every=3500)
    model_params = dict(d_model=250, ntokens=len(dataset.word_map[0]), d_hid=250, nlayers=3, nhead=2, dropout=0.2)

    # ad_gen = Generator(dataset.word_map, model_params, train_params, 'advanced')
    ad_gen: Generator = pickle.load(open(saved_to, 'rb'))
    # ad_gen.train(data_loader)
    plot_graphs(ad_gen.checkpoints['accuracy'], ad_gen.checkpoints['loss'], 'Advanced')

    # pickle.dump(ad_gen, open('cache/advanced/final_generator.pkl', 'wb'))
    # convert_text2midi(ad_gen, 'samples/advanced', amount=20)


def creative_part(amount=10):
    os.makedirs('samples/creative', exist_ok=True)
    tunes = MyDataset.data_prep_text('data_v2.txt')[0]
    ad_gen: Generator = pickle.load(open(f'cache/advanced/final_generator.pkl', 'rb'))

    for tune in tunes:
        if len(tune) <= 120:
            amount -= 1

            original = ' '.join(tune).replace('\n ', '\n')
            s = converter.parse(original)
            s.write('midi', fp=f'samples/creative/original_{amount + 1}.midi')

            new = ad_gen.extend_music(' '.join(tune).replace('\n ', '\n'), 1, 2)
            s = converter.parse(new)
            s.write('midi', fp=f'samples/creative/new_{amount + 1}.midi')

        if amount == 0:
            break


def main():
    dataset = MyDataset(folder_path='data_v2.txt')
    data_loader = DataLoader(dataset, shuffle=True)

    basic_part(dataset, data_loader, epochs=20)
    advanced_part(dataset, data_loader, epochs=20)
    creative_part()


if __name__ == '__main__':
    main()
