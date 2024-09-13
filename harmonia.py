import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, AutoModelForSequenceClassification
import traceback
import torch
import os
import ChordDetection
from collections import Counter

class HarmonIa(object):
    
    def main(self, argv=None):
        # Cria o parser
        parser = argparse.ArgumentParser(description="Processa a música e salva os arquivos")

        # Adiciona os argumentos
        parser.add_argument("-m", "--model", type=str, required=True, help="A pasta do modelo de emoções")
        parser.add_argument("-i", "--input", type=str, required=True, help="O caminho do arquivo de música")
        parser.add_argument("-o", "--output", type=str, required=True, help="A pasta onde os arquivos devem ser salvos")

        # Analisa os argumentos
        args = parser.parse_args()

        # Verifica se o arquivo de música existe
        if not os.path.isfile(args.input):
            print(f"O arquivo {args.input} não existe.")
            exit(1)

        # Verifica se o diretório de saída existe
        if not os.path.isdir(args.output):
            print(f"O diretório {args.output} não existe.")
            exit(1)
            
        if not os.path.isdir(args.model):
            print(f"O diretório {args.model} não existe.")
            exit(1)
        
        self.wav_song = args.input[:-3]+'wav'
        self.song_name = args.input.split('/')[-1].replace('+', ' ')[:-4]
        self.model_path = args.model
        
        
        self.feels = {
            "0": "happy",
            "1": "sad",
            "2": "relaxing",
            "3": "angry",
            "4": "neutral"
        }
        self.feels_inverted = {v: k for k, v in self.feels.items()}
        
        self.call_detect()
        self.predict()
        self.plot_chords(0, 10)
        self.create_list_chords()
        
        self.call_emotions()
        self.predict_emotions()
        self.mean_emotions()
        self.one_hot_preds = self.one_hot(np.array([int(self.feels_inverted[i]) for i in self.feels_result]), len(self.feels))
        self.one_hot_preds = self.one_hot_preds.transpose()
        self.visualize_feels(0, 10)
        
    def call_detect(self):
        self.DC = ChordDetection.DetectChords()
        self.DC.build_cnn_extractor()
        self.DC.build_crf()
        
        self.DC.initialize_chord_axis()
        self.DC.wav_song = self.wav_song
        self.DC.song_name = self.song_name
        
    def predict(self):
        self.predictions_shape = self.DC.predict_seq()
        print(f'Song ends on {self.predictions_shape[1]//10}th second. Keep this in mind when setting time interval to visualize!')

    def create_list_chords(self):
        keys = np.array(list(self.DC.chords.keys()))
        self.list_chords = []
        for i in self.DC.one_hot_preds[:, :].T:
            self.list_chords.append(keys[np.argmax(i)])
            
    def plot_chords(self, start, end):

        plt.rcParams.update({'font.size': 20, 'xtick.major.pad': 15, 
                     'ytick.major.pad': 40, 'axes.titlepad': 15,
                     'xtick.bottom': False, 'ytick.left': False, 'figure.figsize': (70, 25)})
        self.DC.visualize(start, end)
        print(f'Chords for {self.song_name} saved!')
    
    def call_emotions(self):
        # Caminho para a pasta onde você salvou o modelo e o tokenizer
        # Carregar o tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        # Carregar o modelo
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
    
    def predict_emotions(self):
        phrase = ''
        self.feels_result = []
        for idx, chord in enumerate(self.list_chords):
            phrase += chord + ' '
            if(idx % 20 == 0 or idx == len(self.list_chords)-1):
                # 1. Tokenizar a entrada

                tokens = self.tokenizer.encode_plus(phrase, return_tensors="pt")
                encoding = {k: v.to(self.model.device) for k,v in tokens.items()}
                # 2. Fazer a previsão 
                with torch.no_grad():
                    outputs = self.model(**encoding)

                # O output é uma tupla onde o primeiro item é a saída do modelo.
                # O tamanho do output será [1, número de tokens, tamanho do embedding]
                predicted = outputs

                logits = outputs.logits
                feelin_index = np.argmax(logits)
                self.feels_result.append(self.feels[str(feelin_index.item())])
                phrase = ''
    
    def mean_emotions(self):
        counter = Counter(self.feels_result)
        palavra_mais_comum = counter.most_common(1)[0][0]
        print(palavra_mais_comum)
    
    def one_hot(self, class_ids, num_classes):
        class_ids = class_ids.astype('int32')
        oh = np.zeros((len(class_ids), num_classes), dtype=np.int32)
        oh[np.arange(len(class_ids)), class_ids] = 1

        assert (oh.argmax(axis=1) == class_ids).all()
        assert (oh.sum(axis=1) == 1).all()

        return oh
    
    def visualize_feels(self, start, end):
        if(start == 0):
            nstart = 2
        else:
            nstart = start*2
        cmap = sns.dark_palette('purple', as_cmap=True)
        chart = sns.heatmap(self.one_hot_preds[:,start:end], cmap=cmap, xticklabels=np.arange(nstart, end*2+2, 2), yticklabels=np.array(list(self.feels.values())), linewidths=.03, cbar=False)

        chart.set_xticklabels(
            chart.get_xticklabels(), 
            rotation=-45, 
            horizontalalignment='right',
            fontweight='light',
            fontsize='large',
            )

        chart.set_yticklabels(
            chart.get_yticklabels(), 
            rotation=0,
            horizontalalignment='center',
            fontweight='light',
            fontsize='xx-large',
            )

        chart.set_title(f'Feels for {self.song_name}', fontsize=70)
        chart.set_ylabel('Feels', fontsize=50)
        chart.set_xlabel('Seconds', fontsize=50)

        plt.tight_layout()
        plt.savefig(f'{self.song_name}_emotion_{start*2}_{end*2}.svg')
        print(f'Feels for {self.song_name} saved!')
        
def handle(): # pragma: no cover
    """
    Main program execution handler.
    """
    try:
        harmonia = HarmonIa()
        harmonia.main()
    except (KeyboardInterrupt, SystemExit): # pragma: no cover
        return
    except Exception as e:
        print(e)
        traceback.print_exc()
        
if __name__ == "__main__":
    handle()