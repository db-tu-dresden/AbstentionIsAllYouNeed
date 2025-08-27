from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from transformers import pipeline
from tqdm import tqdm
#from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader


import umap.umap_ as umap
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


from scipy.special import softmax
from scipy.stats import entropy

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import gc
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoConfig
from transformers import AutoModel
device = "cuda:0" if torch.cuda.is_available() else "cpu"

import pickle
import time

from transformers import logging
logging.set_verbosity_error()

from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

from datasets import load_dataset, DatasetDict, Dataset
from dependencies.basic_functions import *

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='dataset auswahl')
    parser.add_argument('--dataset', type=str, help='dataset', required=True)
    return parser.parse_args()


def run_all(Dataset_name):
    from datasets import load_dataset
    if Dataset_name == "Spam_text":
        ds = load_dataset("SetFit/enron_spam") #'text','subject' 'label'   --->subject nutzen 70% acc, 96 bei text
        #ds = load_dataset("SalehAhmad/Spam-Ham") #['label', 'sms_text'] #like 99.6% accuracy nicht nutzen
        filtered_dataset_train = ds['train'].select(range(10000))
        X_Dataset_train = filtered_dataset_train["text"]
        match_vector_labels_train = filtered_dataset_train['label']
        filtered_dataset_test = ds['test'].select(range(2000))
        X_Dataset_test = filtered_dataset_test["text"]
        match_vector_labels_test = filtered_dataset_test['label']
        
        X_Dataset = {"train": X_Dataset_train,"test": X_Dataset_test}
        match_vector_labels = {"train": match_vector_labels_train,"test" : match_vector_labels_test}
        
        model_name = "cfdd/spam_text"
        max_length_embedding = 128
        Dataset_name = "Spam_text"
        Classification_Experiments = True
        MatchVector_Embedding = CLF_MatchVector_Embedding

    elif Dataset_name == "TriviaQA":
        model_name = None
        max_length_embedding = 368
        Dataset_name = "TriviaQA_Base_FULL" #Trivia QA mit Context
        Classification_Experiments = False
        
        speicherort = "./dependencies/"+Dataset_name+".pkl"
        with open(speicherort, 'rb') as f:
            phi_dict = pickle.load(f)   
        
        Dataset_name = "Base_Phi_TriviaQA_GLUE"
        
        match_vector = phi_dict["match_vector"]
        match_vector_time = phi_dict["match_vector_time"]
        X_Dataset = {"train": [val[-1]["content"] for val in phi_dict["prompt"]["train"][:10000]], #letzte User Nachricht
                     "test": [val[-1]["content"] for val in phi_dict["prompt"]["val"][:2000]]} #10,2k maximal, meistens weniger aber
        
        
        match_vector_labels = phi_dict["match_vector_labels"]
        match_vector_Original = match_vector
        MatchVector_Embedding = None
        
        # Rename 'val' to 'test'
        match_vector["test"] = match_vector.pop("val")
        match_vector_time["test"] = match_vector_time.pop("val")
        match_vector_labels["test"] = match_vector_labels.pop("val")
        
        Classification_Dict = {"dropout_agreement_mv":phi_dict["agreement_mv"],"dropout_agreement_dp":phi_dict["agreement_dp"]}
        
        len(X_Dataset["train"]),len(X_Dataset["test"])
        
    elif Dataset_name == "Spam_text_OOD":
        ds = load_dataset("SetFit/enron_spam") #'text','subject' 'label'   --->subject nutzen 70% acc, 96 bei text
        #ds = load_dataset("SalehAhmad/Spam-Ham") #['label', 'sms_text'] #like 99.6% accuracy nicht nutzen
        filtered_dataset_train = ds['train'].select(range(10000))
        X_Dataset_train = filtered_dataset_train["text"]
        match_vector_labels_train = filtered_dataset_train['label']
        filtered_dataset_test = ds['test'].select(range(2000))
        X_Dataset_test = filtered_dataset_test["text"]
        match_vector_labels_test = filtered_dataset_test['label']
        
        X_Dataset = {"train": X_Dataset_train,"test": X_Dataset_test}
        match_vector_labels = {"train": match_vector_labels_train,"test" : match_vector_labels_test}
        
        model_name = "mshenoda/roberta-spam"
        max_length_embedding = 128
        Dataset_name = "Spam_text_OOD"
        Classification_Experiments = True
        MatchVector_Embedding = CLF_MatchVector_Embedding
        
    elif Dataset_name == "Spam":
        ds = load_dataset("SetFit/enron_spam") #'text','subject' 'label'   --->subject nutzen 70% acc, 96 bei text
        #ds = load_dataset("SalehAhmad/Spam-Ham") #['label', 'sms_text'] #like 99.6% accuracy nicht nutzen
        filtered_dataset_train = ds['train'].select(range(10000))
        X_Dataset_train = filtered_dataset_train["subject"]
        match_vector_labels_train = filtered_dataset_train['label']
        filtered_dataset_test = ds['test'].select(range(2000))
        X_Dataset_test = filtered_dataset_test["subject"]
        match_vector_labels_test = filtered_dataset_test['label']
        
        X_Dataset = {"train": X_Dataset_train,"test": X_Dataset_test}
        match_vector_labels = {"train": match_vector_labels_train,"test" : match_vector_labels_test}
        
        model_name = "cfdd/spam_header"
        max_length_embedding = 128
        Dataset_name = "Spam"
        Classification_Experiments = True
        MatchVector_Embedding = CLF_MatchVector_Embedding
        
    elif Dataset_name == "Spam_OOD":
        ds = load_dataset("SetFit/enron_spam") #'text','subject' 'label'   --->subject nutzen 70% acc, 96 bei text
        #ds = load_dataset("SalehAhmad/Spam-Ham") #['label', 'sms_text'] #like 99.6% accuracy nicht nutzen
        filtered_dataset_train = ds['train'].select(range(10000))
        X_Dataset_train = filtered_dataset_train["subject"]
        match_vector_labels_train = filtered_dataset_train['label']
        filtered_dataset_test = ds['test'].select(range(2000))
        X_Dataset_test = filtered_dataset_test["subject"]
        match_vector_labels_test = filtered_dataset_test['label']
        
        X_Dataset = {"train": X_Dataset_train,"test": X_Dataset_test}
        match_vector_labels = {"train": match_vector_labels_train,"test" : match_vector_labels_test}
        
        model_name = "mshenoda/roberta-spam"
        max_length_embedding = 128
        Dataset_name = "Spam_OOD"
        Classification_Experiments = True
        MatchVector_Embedding = CLF_MatchVector_Embedding
    #----------------------------------------------------------------------------

    elif Dataset_name == "Sentiment":
        ds = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset")    #'text', 'label', 'sentiment'
        
        dataset= ds['train'].select(range(10001)) #==10000 nach Filter
        mask = [i for i in range(len(dataset)) if i != 5639]
        filtered_dataset = dataset.select(mask)
        
        X_Dataset = {"train": filtered_dataset["text"],"test": ds["test"].select(range(2000))["text"]}
        match_vector_labels = {"train": filtered_dataset["label"],"test" : ds["test"].select(range(2000))["label"]}
        
        model_name = "cfdd/Sentiment"
        max_length_embedding = 128
        Dataset_name = "Sentiment"
        Classification_Experiments = True
        MatchVector_Embedding = CLF_MatchVector_Embedding
    elif Dataset_name == "Sentiment_OOD":
        ds = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset")    #'text', 'label', 'sentiment'
        
        dataset= ds['train'].select(range(10001)) #==10000 nach Filter
        mask = [i for i in range(len(dataset)) if i != 5639]
        filtered_dataset = dataset.select(mask)
        
        X_Dataset = {"train": filtered_dataset["text"],"test": ds["test"].select(range(2000))["text"]}
        match_vector_labels = {"train": filtered_dataset["label"],"test" : ds["test"].select(range(2000))["label"]}
        
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        max_length_embedding = 128
        Dataset_name = "Sentiment_OOD"
        Classification_Experiments = True
        MatchVector_Embedding = CLF_MatchVector_Embedding

    elif Dataset_name == "SST2":
        Dataset_name = "sst2"
        task = Dataset_name
        datasets = {}
        datasets[task] = load_dataset("glue", task)
        
        X_Dataset = {"train": datasets[task]["train"].select(range(10000))["sentence"],"test": datasets[task]["validation"].select(range(872))["sentence"]}
        match_vector_labels = {"train": datasets[task]["train"].select(range(10000))["label"],"test" :datasets[task]["validation"].select(range(872))["label"]}
        
        model_name = "cfdd/SST2"
        max_length_embedding = 128
        Dataset_name = "SST2"
        Classification_Experiments = True
        MatchVector_Embedding = CLF_MatchVector_Embedding

    elif Dataset_name == "MNLI_matched":
        Dataset_name = "mnli"
        task = Dataset_name
        
        datasets = {}
        datasets[task] = load_dataset("glue", task)
        
        filtered_dataset = datasets[task]['train'].select(range(10000,40000))
        filtered_dataset = pd.DataFrame(filtered_dataset)
        X_Dataset_train = (filtered_dataset['premise'] + ' [SEP] ' + filtered_dataset['hypothesis']).tolist()
        labels_train = filtered_dataset['label']
        
        filtered_dataset_test = datasets[task]['validation_matched'].select(range(2000))
        filtered_dataset_test = pd.DataFrame(filtered_dataset_test)
        X_Dataset_test = (filtered_dataset_test['premise'] + ' [SEP] ' + filtered_dataset_test['hypothesis']).tolist()
        labels_test = filtered_dataset_test['label']
        
        X_Dataset = {"train": X_Dataset_train,"test": X_Dataset_test}
        match_vector_labels = {"train": labels_train,"test" : labels_test}
        
        num_labels = len(set(match_vector_labels["train"]))
        dataset = DatasetDict({
            "train":Dataset.from_dict({"text":X_Dataset["train"],"label":match_vector_labels["train"]}),
            "test":Dataset.from_dict({"text":X_Dataset["test"],"label":match_vector_labels["test"]})
        })
        
        
        
        X_Dataset = {"train": dataset["train"]["text"],"test": dataset["test"]["text"]}
        match_vector_labels = {"train": dataset["train"]["label"],"test" :dataset["test"]["label"]}
        
        model_name = "cfdd/MNLI"
        max_length_embedding = 128
        Dataset_name = "MNLI_Matched"
        Classification_Experiments = True
        MatchVector_Embedding = CLF_MatchVector_Embedding
        
        
    elif Dataset_name == "AG_News":
        ds = load_dataset("fancyzhx/ag_news")
        X_train = ds["train"]["text"][50000:60000]#auf ersten 50k Daten Roberta Modell Trainiert
        Y_train = ds["train"]["label"][50000:60000]
        X_test = ds["test"]["text"][2000:4000] #Irgendwelche 2000 Test Datensätze 
        Y_test = ds["test"]["label"][2000:4000]

        X_Dataset = {"train": X_train,"test": X_test}
        match_vector_labels = {"train": Y_train,"test" : Y_test}

        model_name = "cfdd/roberta-ag-news"#"./roberta-ag-news"
        max_length_embedding = 256#312 #ab 128 quasi alle mit drin
        Dataset_name = "AG_News"
        Classification_Experiments = True
        MatchVector_Embedding = CLF_MatchVector_Embedding
        
    elif Dataset_name == "QA":
        ds = load_dataset("rajpurkar/squad")
        filtered_dataset = ds['train'].shuffle(seed=42).select(range(10000))
        filtered_dataset = pd.DataFrame(filtered_dataset)
        X_Dataset_train = (filtered_dataset['question'] + ' [SEP] ' + filtered_dataset['context']).tolist()
        labels_train = [entry['text'][0] for entry in filtered_dataset['answers']]

        filtered_dataset_test = ds['validation'].shuffle(seed=42).select(range(2000))
        filtered_dataset_test = pd.DataFrame(filtered_dataset_test)
        X_Dataset_test = (filtered_dataset_test['question'] + ' [SEP] ' + filtered_dataset_test['context']).tolist()
        labels_test = [entry['text'][0] for entry in filtered_dataset_test['answers']]

        X_Dataset = {"train": X_Dataset_train,"test": X_Dataset_test}
        match_vector_labels = {"train": labels_train,"test" : labels_test}


        #model_name = "deepset/tinyroberta-squad2"
        model_name = "deepset/roberta-base-squad2"
        max_length_embedding = 386
        Dataset_name = "Question Answering"
        Classification_Experiments = False
        MatchVector_Embedding = QA_MatchVector_Embedding
        
    elif Dataset_name == "Regression":
        ds = load_dataset("yashraizad/yelp-open-dataset-top-reviews")
        X_train = ds["train"]["text"][20000:30000]#auf ersten 10k Daten Roberta Modell Trainiert
        Y_train = ds["train"]["stars"][20000:30000]
        X_test = ds["train"]["text"][30000:32000]
        Y_test = ds["train"]["stars"][30000:32000]

        X_Dataset = {"train": X_train,"test": X_test}
        match_vector_labels = {"train": Y_train,"test" : Y_test}
        model_name = "cfdd/roberta_regression"
        max_length_embedding = 128
        Dataset_name = "Regression"
        Classification_Experiments = False
        MatchVector_Embedding = Regression_MatchVector_Embedding
        
    elif Dataset_name =="TSM":
        with open('./dependencies/TimeSeriesDataset.pkl', 'rb') as f:
            TimeSeriesDataset = pickle.load(f)
        
        X_Dataset = TimeSeriesDataset["X_Dataset"]
        match_vector_labels = TimeSeriesDataset["match_vector_labels"]
            
        model_name = "cfdd/roberta_timeseries"
        max_length_embedding = 32
        Dataset_name = "Time Series Regression"
        Classification_Experiments = False
        MatchVector_Embedding = Regression_MatchVector_Embedding
        
    elif Dataset_name == "Transformation_Spellcheck":
        df = pd.read_csv('./dependencies/typo_pairs.csv')# Selbst gemachter Datensatz
        inputs = list(df['Text'])  # Misspelled words from your dataset
        targets = list(df['Label'])  # Corrected words (ground truth)

        X_Dataset = {"train": inputs[:10000],"test": inputs[10000:12000]}
        match_vector_labels = {"train": targets[:10000],"test": targets[10000:12000]}


        model_name = "oliverguhr/spelling-correction-english-base"
        max_length_embedding = 64 #Dataset ist immer nur ein Satz, also 64 schon groß
        Dataset_name = "Transformation"
        Classification_Experiments = False
        MatchVector_Embedding = Transformation_MatchVector_Embedding
        
    #####AB HIER PHI RUNS, einmal Finetuned, rest Base (Meiste Zero shot)
    elif Dataset_name =="AG_News_PHI":
        
        from datasets import load_dataset
        
        ds = load_dataset("fancyzhx/ag_news")
        X_train = ds["train"]["text"][50000:60000]#auf ersten 50k Daten Roberta Modell Trainiert
        Y_train = ds["train"]["label"][50000:60000]
        X_test = ds["test"]["text"][2000:4000] #Irgendwelche 2000 Test Datensätze 
        Y_test = ds["test"]["label"][2000:4000]
        
        X_Dataset = {"train": X_train,"test": X_test}
        #match_vector_labels = {"train": Y_train,"test" : Y_test}
        model_name = None
        max_length_embedding = 256
        Dataset_name = "AG_News_PHI"
        Classification_Experiments = True
        
        
        with open('./dependencies/Phi_results.pkl', 'rb') as f:
            phi_dict = pickle.load(f)   
        
        match_vector = phi_dict["match_vector"]
        match_vector_time = phi_dict["match_vector_time"]
        embeddings_base = phi_dict["embeddings_base"]
        
        softmax_base_df = phi_dict["softmax"]
        softmax_dropout_mv_df = phi_dict["dropout_dfs"]["softmax_mv"]
        softmax_dropout_dp_df = phi_dict["dropout_dfs"]["softmax_dp"]
        agreement_dropout_mv_df = phi_dict["dropout_dfs"]["agreement_mv"]
        agreement_dropout_dp_df = phi_dict["dropout_dfs"]["agreement_dp"]
        
        #return_dict = {"match_vector":match_vector,"match_vector_time":match_vector_time,"embeddings_base":embeddings_base,"softmax":softmax_temp_score_df}
        #dropout_dfs = {"softmax_mv":result_softmax_max_mean,"softmax_dp":result_softmax_max_mean_dp,"agreement_mv":agreement,"agreement_dp":agreement_dp}
        
        Phi_Classification_Dict = {"softmax":softmax_base_df,"dropout_softmax_mv":softmax_dropout_mv_df,"dropout_softmax_dp":softmax_dropout_dp_df,
                                   "dropout_agreement_mv":agreement_dropout_mv_df,"dropout_agreement_dp":agreement_dropout_dp_df}
        
        
        match_vector_labels = None
        match_vector_Original = match_vector

    elif Dataset_name == "Base_Phi_Sentiment":
        model_name = None
        max_length_embedding = 256
        Dataset_name = "Base_Phi_Sentiment"
        Classification_Experiments = False
        
        speicherort = "./dependencies/"+Dataset_name+".pkl"
        with open(speicherort, 'rb') as f:
            phi_dict = pickle.load(f)   
        
        match_vector = phi_dict["match_vector"]
        match_vector_time = phi_dict["match_vector_time"]
        X_Dataset = {"train": [val[-1]["content"] for val in phi_dict["prompt"]["train"]], #letzte User Nachricht
                     "test": [val[-1]["content"] for val in phi_dict["prompt"]["test"]]}
        
        
        match_vector_labels = phi_dict["match_vector_labels"]
        match_vector_Original = match_vector

        # Rename 'val' to 'test'
        match_vector["test"] = match_vector.pop("val")
        match_vector_time["test"] = match_vector_time.pop("val")
        Classification_Dict = {"dropout_agreement_mv":phi_dict["agreement_mv"],"dropout_agreement_dp":phi_dict["agreement_dp"]}

    elif Dataset_name == "Base_Phi_AG_News":
        model_name = None
        max_length_embedding = 256
        Dataset_name = "Base_Phi_AG_News"
        Classification_Experiments = False
        
        speicherort = "./dependencies/"+Dataset_name+".pkl"
        with open(speicherort, 'rb') as f:
            phi_dict = pickle.load(f)   
        
        
        match_vector = phi_dict["match_vector"]
        match_vector_time = phi_dict["match_vector_time"]
        X_Dataset = {"train": [val[-1]["content"] for val in phi_dict["prompt"]["train"]], #letzte User Nachricht
                     "test": [val[-1]["content"] for val in phi_dict["prompt"]["test"]]}
        
        
        match_vector_labels = phi_dict["match_vector_labels"]
        match_vector_Original = match_vector
        MatchVector_Embedding = None

        # Rename 'val' to 'test'
        match_vector["test"] = match_vector.pop("val")
        match_vector_time["test"] = match_vector_time.pop("val")
        Classification_Dict = {"dropout_agreement_mv":phi_dict["agreement_mv"],"dropout_agreement_dp":phi_dict["agreement_dp"]}

    elif Dataset_name == "Base_Phi_Spam":
        model_name = None
        max_length_embedding = 256
        Dataset_name = "Base_Phi_Spam"
        Classification_Experiments = False
        
        speicherort = "./dependencies/"+Dataset_name+".pkl"
        with open(speicherort, 'rb') as f:
            phi_dict = pickle.load(f)   
        
        match_vector = phi_dict["match_vector"]
        match_vector_time = phi_dict["match_vector_time"]
        X_Dataset = {"train": [val[-1]["content"] for val in phi_dict["prompt"]["train"]], #letzte User Nachricht
                     "test": [val[-1]["content"] for val in phi_dict["prompt"]["test"]]}
        
        
        match_vector_labels = phi_dict["match_vector_labels"]
        match_vector_Original = match_vector
        MatchVector_Embedding = None
        
        # Rename 'val' to 'test'
        match_vector["test"] = match_vector.pop("val")
        match_vector_time["test"] = match_vector_time.pop("val")
        Classification_Dict = {"dropout_agreement_mv":phi_dict["agreement_mv"],"dropout_agreement_dp":phi_dict["agreement_dp"]}
        
    elif Dataset_name == "Base_Phi_Spam_text":
        model_name = None
        max_length_embedding = 256
        Dataset_name = "Base_Phi_Spam_text"
        Classification_Experiments = False
        
        speicherort = "./dependencies/"+Dataset_name+".pkl"
        with open(speicherort, 'rb') as f:
            phi_dict = pickle.load(f)   
        
        match_vector = phi_dict["match_vector"]
        match_vector_time = phi_dict["match_vector_time"]
        X_Dataset = {"train": [val[-1]["content"] for val in phi_dict["prompt"]["train"]], #letzte User Nachricht
                     "test": [val[-1]["content"] for val in phi_dict["prompt"]["test"]]}
        
        
        match_vector_labels = phi_dict["match_vector_labels"]
        match_vector_Original = match_vector
        MatchVector_Embedding = None

        # Rename 'val' to 'test'
        match_vector["test"] = match_vector.pop("val")
        match_vector_time["test"] = match_vector_time.pop("val")
        Classification_Dict = {"dropout_agreement_mv":phi_dict["agreement_mv"],"dropout_agreement_dp":phi_dict["agreement_dp"]}

    
    elif Dataset_name == "Base_Phi_QA_4Shot":
        model_name = None
        max_length_embedding = 386
        Dataset_name = "Base_Phi_QA_4Shot"
        Classification_Experiments = False
        
        speicherort = "./dependencies/"+Dataset_name+".pkl"
        with open(speicherort, 'rb') as f:
            phi_dict = pickle.load(f)   
        
        match_vector = phi_dict["match_vector"]
        match_vector_time = phi_dict["match_vector_time"]
        X_Dataset = {"train": [val[-1]["content"] for val in phi_dict["prompt"]["train"]], #letzte User Nachricht
                     "test": [val[-1]["content"] for val in phi_dict["prompt"]["test"]]}
        
        
        match_vector_labels = phi_dict["match_vector_labels"]
        match_vector_Original = match_vector
        MatchVector_Embedding = None

        # Rename 'val' to 'test'
        match_vector["test"] = match_vector.pop("val")
        match_vector_time["test"] = match_vector_time.pop("val")
        Classification_Dict = {"dropout_agreement_mv":phi_dict["agreement_mv"],"dropout_agreement_dp":phi_dict["agreement_dp"]}

    elif Dataset_name == "Base_Phi_Regression":
        model_name = None
        max_length_embedding = 256
        Dataset_name = "Base_Phi_Regression"
        Classification_Experiments = False
        
        speicherort = "./dependencies/"+Dataset_name+".pkl"
        with open(speicherort, 'rb') as f:
            phi_dict = pickle.load(f)   
        
        match_vector = phi_dict["match_vector"]
        match_vector_time = phi_dict["match_vector_time"]
        X_Dataset = {"train": [val[-1]["content"] for val in phi_dict["prompt"]["train"]], #letzte User Nachricht
                     "test": [val[-1]["content"] for val in phi_dict["prompt"]["test"]]}
        
        
        match_vector_labels = phi_dict["match_vector_labels"]
        match_vector_Original = match_vector
        MatchVector_Embedding = None

        # Rename 'val' to 'test'
        match_vector["test"] = match_vector.pop("val")
        match_vector_time["test"] = match_vector_time.pop("val")
        Classification_Dict = {"dropout_agreement_mv":phi_dict["agreement_mv"],"dropout_agreement_dp":phi_dict["agreement_dp"]}

    elif Dataset_name == "Base_Phi_TimeSeries":
        model_name = None
        max_length_embedding = 512
        Dataset_name = "Base_Phi_TimeSeries"
        Classification_Experiments = False
        
        speicherort = "./dependencies/"+Dataset_name+".pkl"
        with open(speicherort, 'rb') as f:
            phi_dict = pickle.load(f)   
        
        match_vector = phi_dict["match_vector"]
        match_vector_time = phi_dict["match_vector_time"]
        X_Dataset = {"train": [val[-1]["content"] for val in phi_dict["prompt"]["train"]], #letzte User Nachricht
                     "test": [val[-1]["content"] for val in phi_dict["prompt"]["test"]]}
        
        
        match_vector_labels = phi_dict["match_vector_labels"]
        match_vector_Original = match_vector
        MatchVector_Embedding = None

        # Rename 'val' to 'test'
        match_vector["test"] = match_vector.pop("val")
        match_vector_time["test"] = match_vector_time.pop("val")
        Classification_Dict = {"dropout_agreement_mv":phi_dict["agreement_mv"],"dropout_agreement_dp":phi_dict["agreement_dp"]}

    
    elif Dataset_name == "Base_Phi_TransformationSpellcheck":
        model_name = None
        max_length_embedding = 256
        Dataset_name = "Base_Phi_TransformationSpellcheck"
        Classification_Experiments = False
        
        speicherort = "./dependencies/"+Dataset_name+".pkl"
        with open(speicherort, 'rb') as f:
            phi_dict = pickle.load(f)   
        
        match_vector = phi_dict["match_vector"]
        match_vector_time = phi_dict["match_vector_time"]
        X_Dataset = {"train": [val[-1]["content"] for val in phi_dict["prompt"]["train"]], #letzte User Nachricht
                     "test": [val[-1]["content"] for val in phi_dict["prompt"]["test"]]}
        
        
        match_vector_labels = phi_dict["match_vector_labels"]
        match_vector_Original = match_vector
        MatchVector_Embedding = None
        
        # Rename 'val' to 'test'
        match_vector["test"] = match_vector.pop("val")
        match_vector_time["test"] = match_vector_time.pop("val")
        
        Classification_Dict = {"dropout_agreement_mv":phi_dict["agreement_mv"],"dropout_agreement_dp":phi_dict["agreement_dp"]}


    elif Dataset_name == "Base_Phi_GLUE_COLA":        
        model_name = None
        max_length_embedding = 256
        Dataset_name = "Base_Phi_GLUE_COLA" #sentence grammatically correct?
        Classification_Experiments = False
        
        speicherort = "./dependencies/"+Dataset_name+".pkl"
        with open(speicherort, 'rb') as f:
            phi_dict = pickle.load(f)   
        
        
        match_vector = phi_dict["match_vector"]
        match_vector_time = phi_dict["match_vector_time"]
        X_Dataset = {"train": [val[-1]["content"] for val in phi_dict["prompt"]["train"][:10000]], #letzte User Nachricht
                     "test": [val[-1]["content"] for val in phi_dict["prompt"]["val"][:2000]]} #10,2k maximal, meistens weniger aber
        
        
        match_vector_labels = phi_dict["match_vector_labels"]
        match_vector_Original = match_vector
        MatchVector_Embedding = None
        
        # Rename 'val' to 'test'
        match_vector["test"] = match_vector.pop("val")
        match_vector_time["test"] = match_vector_time.pop("val")
        match_vector_labels["test"] = match_vector_labels.pop("val")

        Classification_Dict = {"dropout_agreement_mv":phi_dict["agreement_mv"],"dropout_agreement_dp":phi_dict["agreement_dp"]}
        
    elif Dataset_name == "Base_Phi_GLUE_QNLI":        
        model_name = None
        max_length_embedding = 256
        Dataset_name = "Base_Phi_GLUE_QNLI" # whether a sentence answers a given question or not
        Classification_Experiments = False
        
        speicherort = "./dependencies/"+Dataset_name+".pkl"
        with open(speicherort, 'rb') as f:
            phi_dict = pickle.load(f)   
        
        
        match_vector = phi_dict["match_vector"]
        match_vector_time = phi_dict["match_vector_time"]
        X_Dataset = {"train": [val[-1]["content"] for val in phi_dict["prompt"]["train"][:10000]], #letzte User Nachricht
                     "test": [val[-1]["content"] for val in phi_dict["prompt"]["val"][:2000]]} #10,2k maximal, meistens weniger aber
        
        
        match_vector_labels = phi_dict["match_vector_labels"]
        match_vector_Original = match_vector
        MatchVector_Embedding = None
        
        # Rename 'val' to 'test'
        match_vector["test"] = match_vector.pop("val")
        match_vector_time["test"] = match_vector_time.pop("val")
        match_vector_labels["test"] = match_vector_labels.pop("val")
        
        Classification_Dict = {"dropout_agreement_mv":phi_dict["agreement_mv"],"dropout_agreement_dp":phi_dict["agreement_dp"]}        
        
    elif Dataset_name == "Base_Phi_GLUE_QQP":        
        model_name = None
        max_length_embedding = 256
        Dataset_name = "Base_Phi_GLUE_QQP" # are two questions are asking the same thing or not
        Classification_Experiments = False
        
        speicherort = "./dependencies/"+Dataset_name+".pkl"
        with open(speicherort, 'rb') as f:
            phi_dict = pickle.load(f)   
        
        
        match_vector = phi_dict["match_vector"]
        match_vector_time = phi_dict["match_vector_time"]
        X_Dataset = {"train": [val[-1]["content"] for val in phi_dict["prompt"]["train"][:10000]], #letzte User Nachricht
                     "test": [val[-1]["content"] for val in phi_dict["prompt"]["val"][:2000]]} #10,2k maximal, meistens weniger aber
        
        
        match_vector_labels = phi_dict["match_vector_labels"]
        match_vector_Original = match_vector
        MatchVector_Embedding = None
        
        # Rename 'val' to 'test'
        match_vector["test"] = match_vector.pop("val")
        match_vector_time["test"] = match_vector_time.pop("val")
        match_vector_labels["test"] = match_vector_labels.pop("val")

        Classification_Dict = {"dropout_agreement_mv":phi_dict["agreement_mv"],"dropout_agreement_dp":phi_dict["agreement_dp"]}
        
    elif Dataset_name == "Base_Phi_GLUE_RTE":        
        model_name = None
        max_length_embedding = 256
        Dataset_name = "Base_Phi_GLUE_RTE" # if a hypothesis follows from a given premise
        Classification_Experiments = False
        
        speicherort = "./dependencies/"+Dataset_name+".pkl"
        with open(speicherort, 'rb') as f:
            phi_dict = pickle.load(f)   
        
        
        match_vector = phi_dict["match_vector"]
        match_vector_time = phi_dict["match_vector_time"]
        X_Dataset = {"train": [val[-1]["content"] for val in phi_dict["prompt"]["train"][:10000]], #letzte User Nachricht
                     "test": [val[-1]["content"] for val in phi_dict["prompt"]["val"][:2000]]} #10,2k maximal, meistens weniger aber
        
        
        match_vector_labels = phi_dict["match_vector_labels"]
        match_vector_Original = match_vector
        MatchVector_Embedding = None
        
        # Rename 'val' to 'test'
        match_vector["test"] = match_vector.pop("val")
        match_vector_time["test"] = match_vector_time.pop("val")
        match_vector_labels["test"] = match_vector_labels.pop("val")

        Classification_Dict = {"dropout_agreement_mv":phi_dict["agreement_mv"],"dropout_agreement_dp":phi_dict["agreement_dp"]}
        
        
    elif Dataset_name == "Base_Phi_GLUE_SST2":        
        model_name = None
        max_length_embedding = 256
        Dataset_name = "Base_Phi_GLUE_SST2" # classify the sentiment of a given sentence as either Positive or Negative
        Classification_Experiments = False
        
        speicherort = "./dependencies/"+Dataset_name+".pkl"
        with open(speicherort, 'rb') as f:
            phi_dict = pickle.load(f)   
        
        
        match_vector = phi_dict["match_vector"]
        match_vector_time = phi_dict["match_vector_time"]
        X_Dataset = {"train": [val[-1]["content"] for val in phi_dict["prompt"]["train"][:10000]], #letzte User Nachricht
                     "test": [val[-1]["content"] for val in phi_dict["prompt"]["val"][:2000]]} #10,2k maximal, meistens weniger aber
        
        
        match_vector_labels = phi_dict["match_vector_labels"]
        match_vector_Original = match_vector
        MatchVector_Embedding = None
        
        # Rename 'val' to 'test'
        match_vector["test"] = match_vector.pop("val")
        match_vector_time["test"] = match_vector_time.pop("val")
        match_vector_labels["test"] = match_vector_labels.pop("val")

        Classification_Dict = {"dropout_agreement_mv":phi_dict["agreement_mv"],"dropout_agreement_dp":phi_dict["agreement_dp"]}
        
        
    elif Dataset_name == "Base_Phi_GLUE_STSB":        
        model_name = None
        max_length_embedding = 256
        Dataset_name = "Base_Phi_GLUE_STSB" # similarity between two sentences on a scale from 1 to 5
        Classification_Experiments = False
        
        speicherort = "./dependencies/"+Dataset_name+".pkl"
        with open(speicherort, 'rb') as f:
            phi_dict = pickle.load(f)   
        
        
        match_vector = phi_dict["match_vector"]
        match_vector_time = phi_dict["match_vector_time"]
        X_Dataset = {"train": [val[-1]["content"] for val in phi_dict["prompt"]["train"][:10000]], #letzte User Nachricht
                     "test": [val[-1]["content"] for val in phi_dict["prompt"]["val"][:2000]]} #10,2k maximal, meistens weniger aber
        
        
        match_vector_labels = phi_dict["match_vector_labels"]
        match_vector_Original = match_vector
        MatchVector_Embedding = None
        
        # Rename 'val' to 'test'
        match_vector["test"] = match_vector.pop("val")
        match_vector_time["test"] = match_vector_time.pop("val")
        match_vector_labels["test"] = match_vector_labels.pop("val")

        Classification_Dict = {"dropout_agreement_mv":phi_dict["agreement_mv"],"dropout_agreement_dp":phi_dict["agreement_dp"]}
        
        
    elif Dataset_name == "Base_Phi_GLUE_WNLI":        
        model_name = None
        max_length_embedding = 256
        Dataset_name = "Base_Phi_GLUE_WNLI" # determine if the second sentence is entailed by the first
        Classification_Experiments = False
        
        speicherort = "./dependencies/"+Dataset_name+".pkl"
        with open(speicherort, 'rb') as f:
            phi_dict = pickle.load(f)   
        
        
        match_vector = phi_dict["match_vector"]
        match_vector_time = phi_dict["match_vector_time"]
        X_Dataset = {"train": [val[-1]["content"] for val in phi_dict["prompt"]["train"][:10000]], #letzte User Nachricht
                     "test": [val[-1]["content"] for val in phi_dict["prompt"]["val"][:2000]]} #10,2k maximal, meistens weniger aber
        
        
        match_vector_labels = phi_dict["match_vector_labels"]
        match_vector_Original = match_vector
        MatchVector_Embedding = None
        
        # Rename 'val' to 'test'
        match_vector["test"] = match_vector.pop("val")
        match_vector_time["test"] = match_vector_time.pop("val")
        match_vector_labels["test"] = match_vector_labels.pop("val")

        Classification_Dict = {"dropout_agreement_mv":phi_dict["agreement_mv"],"dropout_agreement_dp":phi_dict["agreement_dp"]}
        
        
    elif Dataset_name == "Base_Phi_GLUE_MNLI_MISMATCHED":        
        model_name = None
        max_length_embedding = 256
        Dataset_name = "Base_Phi_GLUE_MNLI" # Does the hypothesis entail the premise, contradict it, or is it neutral?
        Classification_Experiments = False
        
        speicherort = "./dependencies/"+Dataset_name+".pkl"
        with open(speicherort, 'rb') as f:
            phi_dict = pickle.load(f)   
        
        
        match_vector = phi_dict["match_vector"]
        match_vector_time = phi_dict["match_vector_time"]
        X_Dataset = {"train": [val[-1]["content"] for val in phi_dict["prompt"]["train"][:10000]], #letzte User Nachricht
                     "test": [val[-1]["content"] for val in phi_dict["prompt"]["mismatched"][:2000]]} #10,2k maximal, meistens weniger aber
        
        
        match_vector_labels = phi_dict["match_vector_labels"]
        match_vector_Original = match_vector
        MatchVector_Embedding = None
        
        # Rename 'val' to 'test'
        match_vector["test"] = match_vector.pop("mismatched")
        match_vector_time["test"] = match_vector_time.pop("mismatched")
        match_vector_labels["test"] = match_vector_labels.pop("mismatched")
        
        Dataset_name = "Base_Phi_GLUE_MNLI_MISMATCHED"

        Classification_Dict = {"dropout_agreement_mv":phi_dict["agreement_mv_unmatched"],"dropout_agreement_dp":phi_dict["agreement_dp_unmatched"]}
        
        
    elif Dataset_name == "Base_Phi_GLUE_MNLI_MATCHED":        
        model_name = None
        max_length_embedding = 256
        Dataset_name = "Base_Phi_GLUE_MNLI" # Does the hypothesis entail the premise, contradict it, or is it neutral?
        Classification_Experiments = False
        
        speicherort = "./dependencies/"+Dataset_name+".pkl"
        with open(speicherort, 'rb') as f:
            phi_dict = pickle.load(f)   
        
        
        match_vector = phi_dict["match_vector"]
        match_vector_time = phi_dict["match_vector_time"]
        X_Dataset = {"train": [val[-1]["content"] for val in phi_dict["prompt"]["train"][:10000]], #letzte User Nachricht
                     "test": [val[-1]["content"] for val in phi_dict["prompt"]["matched"][:2000]]} #10,2k maximal, meistens weniger aber
        
        
        match_vector_labels = phi_dict["match_vector_labels"]
        match_vector_Original = match_vector
        MatchVector_Embedding = None
        
        # Rename 'val' to 'test'
        match_vector["test"] = match_vector.pop("matched")
        match_vector_time["test"] = match_vector_time.pop("matched")
        match_vector_labels["test"] = match_vector_labels.pop("matched")
        
        Dataset_name = "Base_Phi_GLUE_MNLI_MATCHED"
        
        Classification_Dict = {"dropout_agreement_mv":phi_dict["agreement_mv_matched"],"dropout_agreement_dp":phi_dict["agreement_dp_matched"]}       
        
    elif Dataset_name == "Base_Phi_GLUE_MRPC":        
        model_name = None
        max_length_embedding = 256
        Dataset_name = "Base_Phi_GLUE_MRPC" # if two sentences are paraphrases of each other
        Classification_Experiments = False
        
        speicherort = "./dependencies/"+Dataset_name+".pkl"
        with open(speicherort, 'rb') as f:
            phi_dict = pickle.load(f)   
        
        
        match_vector = phi_dict["match_vector"]
        match_vector_time = phi_dict["match_vector_time"]
        X_Dataset = {"train": [val[-1]["content"] for val in phi_dict["prompt"]["train"][:10000]], #letzte User Nachricht
                     "test": [val[-1]["content"] for val in phi_dict["prompt"]["val"][:2000]]} #10,2k maximal, meistens weniger aber
        
        
        match_vector_labels = phi_dict["match_vector_labels"]
        match_vector_Original = match_vector
        MatchVector_Embedding = None
        
        # Rename 'val' to 'test'
        match_vector["test"] = match_vector.pop("val")
        match_vector_time["test"] = match_vector_time.pop("val")
        match_vector_labels["test"] = match_vector_labels.pop("val")

        Classification_Dict = {"dropout_agreement_mv":phi_dict["agreement_mv"],"dropout_agreement_dp":phi_dict["agreement_dp"]}
    
    else: 
        print("Model/Dataset nicht implementiert, wahrscheinlich falschgeschrieben")
        return ValueError
        

    from datetime import datetime
    print("Start-Time ",datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    if Dataset_name == "AG_News_PHI":
        PHI = True 
        Zero_Shot_Phi = False
    
    elif Dataset_name in (
        "Base_Phi_Sentiment", 
        "Base_Phi_AG_News", 
        "Base_Phi_Spam", 
        "Base_Phi_Spam_text", 
        "Base_Phi_QA_4Shot", 
        "Base_Phi_TransformationSpellcheck",
        "Base_Phi_Regression",
        "Base_Phi_TimeSeries",
    ):
        PHI = False
        Zero_Shot_Phi = True #Keine Classification, keine Base Embeddings, QA ist auch 4shot lol
        #Glue Datasets auf Phi False, Zero shot True, aber unten bearbeitet, wegen uneven Datasize
        
    else: 
        PHI = False
        Zero_Shot_Phi = False
    
    
    
    #Datasize Series Loop, Dataset Loaden notwendig
    
    datasizes = [10000,8000,6000,4000,2000,1000,500] #MIT GRÖßTER SIZE ANFANGEN
    #datasizes = [500]

    dropout_runs,dropout_rate = 10, 0.1

    max_datasize = datasizes[0]
    if "GLUE" in Dataset_name:
        Zero_Shot_Phi = True
        
        train_max_datasize = len(X_Dataset["train"])
        if train_max_datasize == 10000:
            print(f"GLUE EXPERIMENT, STANDART DATASIZE EXPERIMENTS: {datasizes}")
        else:
            max_datasize = train_max_datasize
            datasizes = [max_datasize]
            print(f"GLUE EXPERIMENT, NON STANDART DATASIZE: {max_datasize}, ONLY ONE DATASIZE RUN")
    
    
    combined_inter_df = []
    combined_time_results = {}
    
    
    X_Dataset_Original = X_Dataset
    match_vector_labels_Original = match_vector_labels
    
    
    
    
    if Dataset_name == "Question Answering":
        MatchVector_Embedding = QA_MatchVector_Embedding
    elif Dataset_name == "Spam" or Dataset_name == "Sentiment" or Dataset_name == "Spam_text" or Dataset_name == "AG_News":
        MatchVector_Embedding = CLF_MatchVector_Embedding
        Classification_Experiments = True
    elif Dataset_name == "Regression"or Dataset_name == "Time Series Regression":
        MatchVector_Embedding = Regression_MatchVector_Embedding
    elif Dataset_name == "Transformation":
        MatchVector_Embedding = Transformation_MatchVector_Embedding
    else:
        print("Falscher Datensatz")
    
    roc_list = [] #ROC Datastructure, wird am ende Unified
    for size in datasizes:   
        print("Loop Size", size, " Dataset: ", Dataset_name)
        print("Loop Time ",datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("PHI", PHI, "Phi_zero_shot", Zero_Shot_Phi)
        print("----------------------------") 
        
        if PHI:
            X_Dataset = {"train": X_Dataset_Original["train"][:size] ,"test": X_Dataset_Original["test"]}
            match_vector_time = {"train": match_vector_time["train"]*(size/10000), #Zeitrechnung künstlich approxmiert
                    "test": match_vector_time["test"]} #Test size bleibt gleich, daher keine modifikation
            match_vector = {"train": match_vector_Original["train"][:size],
                    "test": match_vector_Original["test"]}
            embeddings_base = {"train": embeddings_base["train"][:size],
                    "test": embeddings_base["test"]}
        elif Zero_Shot_Phi:
            X_Dataset = {"train": X_Dataset_Original["train"][:size] ,"test": X_Dataset_Original["test"]}
            match_vector_time = {"train": match_vector_time["train"]*(size/max_datasize), #Zeitrechnung künstlich approxmiert
                    "test": match_vector_time["test"]} #Test size bleibt gleich, daher keine modifikation
            match_vector = {"train": match_vector_Original["train"][:size],
                    "test": match_vector_Original["test"]}
            
        else:
            #[:size] -> zum verkleinern Trainset only aber 
            X_Dataset = {"train": X_Dataset_Original["train"][:size] ,"test": X_Dataset_Original["test"]}
            match_vector_labels = {"train": match_vector_labels_Original["train"][:size],"test" : match_vector_labels_Original["test"]}
    
            ## Match Vector Generation
            #Train
            result_dict_train = MatchVector_Embedding(X_Dataset["train"],match_vector_labels["train"],
                                                          model_name,max_length_embedding)
            MatchVectorTime_train = result_dict_train["MatchVectorTime"]
            #Test - Nur Initial machen 
            if size == max(datasizes): 
                result_dict_test = MatchVector_Embedding(X_Dataset["test"],match_vector_labels["test"],
                                                              model_name,max_length_embedding)
                MatchVectorTime_test = result_dict_train["MatchVectorTime"]
    
            match_vector_time = {"train": MatchVectorTime_train,
                                 "test": MatchVectorTime_test}
            match_vector = {"train": result_dict_train["match_vector"],
                            "test": result_dict_test["match_vector"]}
            embeddings_base = {"train":result_dict_train["embeddings_base"],
                               "test":result_dict_test["embeddings_base"]}
        
        if Classification_Experiments and not PHI:
            results = {"train":result_dict_train["results"],
                        "test":result_dict_test["results"]}
            results_logits = {"train":result_dict_train["logits"],
                    "test":result_dict_test["logits"]}
            results_label = {"train":result_dict_train["label"], 
                        "test":result_dict_test["label"]}
            softmax = {"train":result_dict_train["softmax"],
                        "test":result_dict_test["softmax"]}
        
        
        #--------------------------------------------------------------------------------------
        
        #MAIN TEST LOOP
        # Entropy & Softmax Scores - Classic Uncertainty Methode - Benutzt Ergebniss Main Model
        if Classification_Experiments and not PHI:
            Classification_Dict = {"softmax":softmax_score_calc(softmax,match_vector)}
            
            dropout_return_dict = CLF_Dropout_FULL(X_Dataset["test"],match_vector_labels["test"],
                                                             model_name,max_length_embedding,match_vector,
                                                             dropout_runs=dropout_runs,dropout_rate=dropout_rate)
            Classification_Dict["dropout_softmax_mv"] = dropout_return_dict["softmax_mv"]
            Classification_Dict["dropout_softmax_dp"] = dropout_return_dict["softmax_dp"]
    
            Classification_Dict["dropout_agreement_mv"] = dropout_return_dict["agreement_mv"]
            Classification_Dict["dropout_agreement_dp"] = dropout_return_dict["agreement_dp"]
            #"agreement_mv":agreement_match_vector,"agreement_dp"
            
        elif Classification_Experiments and PHI:
            Classification_Dict = Phi_Classification_Dict #Precalculated
    
        elif Zero_Shot_Phi and not Classification_Experiments:
            #Classification Dict bereits reingeladen
            #Sollte nur bei BasePhi Datasets erreicht werden
            print("Keine Classification Experimente, Dropout Precalculated (Base Phi)")
        elif Dataset_name == "Question Answering" or Dataset_name == "Transformation":
            Classification_Dict = {}
            print("Info: Using Frequency dropout ")
            dropout_return_dict = Frequency_dropout_Full(X_Dataset,match_vector_labels,
                                                     model_name,max_length_embedding=368,match_vector=match_vector,
                                                     dropout_runs=10,dropout_rate=0.1,origin_dataset=Dataset_name)
    
            Classification_Dict["dropout_agreement_mv"] = dropout_return_dict["agreement_mv"]
            Classification_Dict["dropout_agreement_dp"] = dropout_return_dict["agreement_dp"]
    
        elif Dataset_name == "Time Series Regression" or Dataset_name == "Regression":
            print("Info: Using Variance Dropout ")
            Classification_Dict = {}
            dropout_return_dict = Variance_dropout_Full(X_Dataset,match_vector_labels,
                                                     model_name,max_length_embedding=128,match_vector=match_vector,
                                                     dropout_runs=10,dropout_rate=0.1,Dataset_name=Dataset_name)
            Classification_Dict["dropout_agreement_mv"] = dropout_return_dict["agreement_mv"]
            Classification_Dict["dropout_agreement_dp"] = dropout_return_dict["agreement_dp"]
            
        else:
            Classification_Dict = None
            print("Keine Classification Experimente")
        
    
        #Embeddings
        #----------------------------------------------------------------------------------
        bert_embeddings = {}
        results_embedding = {}
    
        if Zero_Shot_Phi == False:
            results_embedding["Base"] = run_all_models(embeddings_base, match_vector, printer=False) 
            bert_embeddings["Base"] = (embeddings_base,match_vector_time["train"],match_vector_time["test"])
            
        bert_embeddings["Distilbert"] = Bert_embeddings_full(X_Dataset,max_length_embedding,modelname="distilbert") 
        bert_embeddings["TinyBert"] = Bert_embeddings_full(X_Dataset,max_length_embedding,modelname="tinybert")
        bert_embeddings["Bert"] = Bert_embeddings_full(X_Dataset,max_length_embedding,modelname="bert") 
        bert_embeddings["RoBERTa"] = Bert_embeddings_full(X_Dataset,max_length_embedding,modelname="roberta-base") 
        bert_embeddings["Tf-Idf"] = tfidf_embeddings(X_Dataset,max_features=5000)
        
        for model,data in bert_embeddings.items():
            embeddings, time_train, time_test = data   
            temp_results_embedding = run_all_models(embeddings, match_vector,printer=False)
            temp_results_embedding["Method_Fit_Time"] = time_train
            temp_results_embedding["Method_Test_Time"] = time_test
            results_embedding[model] = temp_results_embedding
            
        
        #Outlier
        #----------------------------------------------------------------------------------
        Outlier_Dict = {}
        if Zero_Shot_Phi == False:
            Outlier_Dict["Base"] = IsoForest_full(bert_embeddings["Base"][0],match_vector)
        Outlier_Dict["Tf-Idf"] = IsoForest_full(bert_embeddings["Tf-Idf"][0],match_vector)
        Outlier_Dict["Distilbert"] = IsoForest_full(bert_embeddings["Distilbert"][0],match_vector)
        Outlier_Dict["TinyBert"] = IsoForest_full(bert_embeddings["TinyBert"][0],match_vector)
        Outlier_Dict["Bert"] = IsoForest_full(bert_embeddings["Bert"][0],match_vector)
        Outlier_Dict["RoBERTa"] = IsoForest_full(bert_embeddings["RoBERTa"][0],match_vector)
    
        #Dummy & Bert Rejector
        #--------------------------------------------------------------------------------
        if Zero_Shot_Phi == False:
            data_logits_dummy = dummy_data(embeddings_base,match_vector)# # Embeddings egal, kann auch nur 1sen sein
        else: data_logits_dummy = dummy_data(bert_embeddings["Tf-Idf"][0],match_vector)
            
        
        bert_rejectors = {}
        bert_rejectors["Distilbert"] = BertRejector(X_Dataset,match_vector,max_length_embedding,modelname="distilbert")
        bert_rejectors["TinyBert"] = BertRejector(X_Dataset,match_vector,max_length_embedding,modelname="tinybert")
        bert_rejectors["Bert"] = BertRejector(X_Dataset,match_vector,max_length_embedding,modelname="bert")
        bert_rejectors["RoBERTa"] = BertRejector(X_Dataset,match_vector,max_length_embedding,modelname="roberta-base")
        
        #RunTime Results
        time_result = timer(Classification_Dict,match_vector_time,results_embedding,Outlier_Dict,data_logits_dummy,bert_rejectors)
        combined_time_results[size] = time_result
        #,data_logits_tfidf_dict,uncertainty_base,,uncertainty_tfidf
        
        #----------------------------------------------------------------------------------------------------    
        rest_dfs = [] #Accuracy_Coverage Stuff 
        if Classification_Experiments:
            df_0a,r_0a = create_pandas_frame(Classification_Dict["softmax"],Dataset_name,"Softmax",size,"generic")
            df_0c,r_0c = create_pandas_frame(Classification_Dict["dropout_softmax_mv"],Dataset_name,"Dropout Softmax MV",size,"generic")
            df_0d,r_0d = create_pandas_frame(Classification_Dict["dropout_softmax_dp"],Dataset_name,"Dropout Softmax DP",size,"generic")
            df_0e,r_0e = create_pandas_frame(Classification_Dict["dropout_agreement_mv"],Dataset_name,"Dropout Agreement MV",size,"generic")
            df_0f,r_0f = create_pandas_frame(Classification_Dict["dropout_agreement_dp"],Dataset_name,"Dropout Agreement DP",size,"generic")
            rest_dfs = [df_0a,df_0c,df_0d,df_0e,df_0f] 
            roc_list.extend([r_0a,r_0c,r_0d,r_0e,r_0f])
        elif set(Classification_Dict.keys()) == {'dropout_agreement_mv', 'dropout_agreement_dp'}:
            df_0e,r_0e = create_pandas_frame(Classification_Dict["dropout_agreement_mv"],Dataset_name,"Dropout Agreement MV",size,"generic")
            df_0f,r_0f = create_pandas_frame(Classification_Dict["dropout_agreement_dp"],Dataset_name,"Dropout Agreement DP",size,"generic")
            rest_dfs = [df_0e,df_0f] 
            roc_list.extend([r_0e,r_0f])
            
    
        df_embeddings = []
        for model, model_data in results_embedding.items():
            temp_df_a, temp_roc_a = create_pandas_frame(model_data["svc"],Dataset_name,f"{model} Embedding",size,"SVM")
            temp_df_b, temp_roc_b = create_pandas_frame(model_data["rf"],Dataset_name,f"{model} Embedding",size,"Random Forest")
            temp_df_c, temp_roc_c = create_pandas_frame(model_data["lr"],Dataset_name,f"{model} Embedding",size,"Logistic Regression")
            temp_df_d, temp_roc_d = create_pandas_frame(model_data["nn"],Dataset_name,f"{model} Embedding",size,"Simple NN")
            df_embeddings.extend([temp_df_a,temp_df_b,temp_df_c,temp_df_d])
            roc_list.extend([temp_roc_a,temp_roc_b,temp_roc_c,temp_roc_d])
            
            #df_embeddings.append(create_pandas_frame(model_data["svc"],Dataset_name,f"{model} Embedding",size,"SVM"))
            #df_embeddings.append(create_pandas_frame(model_data["rf"],Dataset_name,f"{model} Embedding",size,"Random Forest"))
            #df_embeddings.append(create_pandas_frame(model_data["lr"],Dataset_name,f"{model} Embedding",size,"Logistic Regression"))
            #df_embeddings.append(create_pandas_frame(model_data["nn"],Dataset_name,f"{model} Embedding",size,"Simple NN"))
        
    
        df_4,r_4 = create_pandas_frame(data_logits_dummy,Dataset_name,"Most Frequent Class",size,"generic")
        rest_dfs.append(df_4)
        roc_list.append(r_4)
        
        df_outlier = []
        for name, data in Outlier_Dict.items():
            temp_df, temp_roc = create_pandas_frame(data,Dataset_name,f"{name} Outlier",size,"generic")
            df_outlier.append(temp_df)
            roc_list.append(temp_roc)
            #df_outlier.append(create_pandas_frame(data,Dataset_name,f"{name} Outlier",size,"generic"))
        
        df_rejector = []
        for rejector, rejector_data in bert_rejectors.items():
            temp_df, temp_roc = create_pandas_frame(rejector_data,Dataset_name,f"{rejector} Rejector",size,"generic")
            df_rejector.append(temp_df)
            roc_list.append(temp_roc)
            #df_rejector.append(create_pandas_frame(rejector_data,Dataset_name,f"{rejector} Rejector",size,"generic"))
    
        combined_df = pd.concat(rest_dfs+df_embeddings+df_rejector+df_outlier, ignore_index=True)
        column_order = ['Dataset', 'Data_size', 'Method','Classifier', 'Accuracy', 'Coverage']
        result_df = combined_df[column_order]
    
    
        #Extra Spalte Method_Classifer für einfachere Berechnungen
        #result_df['Method_Classifier'] = result_df['Method'].astype(str) + ' ' + result_df['Classifier'].astype(str)
        result_df.loc[:, 'Method_Classifier'] = result_df['Method'].astype(str) + ' ' + result_df['Classifier'].astype(str)
        result_df = result_df[['Method_Classifier','Dataset', 'Data_size', 'Method','Classifier', 'Accuracy', 'Coverage']]#.drop_duplicates()
        #----------------------------------------------------------------------------------------------------
        
        
        combined_inter_df.append(result_df)
        
    
    full_df = pd.concat(combined_inter_df, ignore_index=True)
    ROC_DICT = {Dataset_name:unify_dicts(roc_list)}
    
    #Merged Time Dict zu einem Dataframe
    for size, df in combined_time_results.items():
        df['Data_size'] = size
    time_df = pd.concat(combined_time_results.values(), ignore_index=True)
    
    print("FERTIG")
    print("End-Time ",datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if Dataset_name == "Question Answering":
        Dataset_name = "QA"
    if Dataset_name == "Time Series Regression":
        Dataset_name = "TSReg"
    
    dateiname1 = './results/'+'results_' + Dataset_name + '_full.pkl'
    final_result_dict = {"full_df":full_df,"time_df":time_df, "ROC_DICT":ROC_DICT}
    
    with open(dateiname1, 'wb') as f:
        pickle.dump(final_result_dict, f)

        

if __name__ == '__main__':
    args = parse_args()
    print(f"Startet mit {args.dataset} Dataset")
    run_all(args.dataset)
