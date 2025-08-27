

import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoModelForSequenceClassification
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.special import softmax
#from tqdm import tqdm
from tqdm.auto import tqdm

from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModel, AutoModelForSequenceClassification

import gc
import time
from sklearn.linear_model import SGDClassifier #SVC ersatz -> SVM Linear 

def most_common_answer(answers):

    if not answers: ###answer list ist leer
        return None,0,0
    # Count occurrences of each unique answer
    answer_counts = Counter(answers)
    
    # Find the most common answer and its count
    most_common_answer, most_common_count = answer_counts.most_common(1)[0]

    # Calculate certainty percentage
    total_answers = len(answers)
    certainty_percentage = (most_common_count / total_answers) * 100

    return most_common_answer, most_common_count, certainty_percentage

#Accuracy von Truth Klasse für Rejector Prediction, 
# Input: y_pred, match_vector
# Output: Accuracy
def precisionprinter_bool(y_test,y_pred,printer=True):                                  
    report = classification_report(y_test, y_pred, digits=5, output_dict=True, zero_division=1)
    precision_class_1 = report['True']['precision']
    if printer:
        print(f"Precision class 1: {precision_class_1:.5f}","Coverage:", np.sum(np.array(y_pred))/len(y_pred))
    return precision_class_1,np.sum(np.array(y_pred))/len(y_pred)

#Input Scores(Logits) und y_test(BOOL! MATCHVECTOR ZB)
#Output Coverage-Accuracy Linie
#NEU,  gibt noch ROC&AUC score dazu mit
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def cov_acc_df_creator(scores, y_test, delete=-1, printer=False):
    acc_list_logits = []
    cov_list_logits = []    
    logits_range = np.quantile(scores, np.linspace(0, 1, 100 + 1))
    
    # For ROC calculation
    fpr_list = []
    tpr_list = []
    
    for threshold in logits_range:
        y_pred = (scores >= threshold)  # Convert logits to binary predictions
        acc, cov = precisionprinter_bool(y_test, y_pred, printer=printer)
        acc_list_logits.append(acc)
        cov_list_logits.append(cov)
        


    # Remove last value as it leads to zero accuracy and coverage
    acc_list_logits = acc_list_logits[:delete]
    cov_list_logits = cov_list_logits[:delete]

    # Calculate ROC curve data
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)
    """
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
    """
    # Return accuracy-coverage data and ROC AUC
    accuracy_coverage = {
        'Accuracy': acc_list_logits,
        'Coverage': cov_list_logits
    }

    return_dict = {
        "accuracy_coverage":accuracy_coverage,
        "FPR": fpr,
        "TPR": tpr,
        "roc_auc": roc_auc
    }
    
    return return_dict #früher nur accuracy_coverage gesendet, erreichbar mit return_dict["accuracy_coverage"]



from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Simple Neural Network
class SimpleNNBinary(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleNNBinary, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)  # Single output for binary classification (logit)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)  # Logits (before sigmoid)
        return out
    


def run_simplenn_binary(X_train, X_test, y_train, y_test, hidden_size=128, epochs=100, learning_rate=0.001):
    # Split dataset into training and testing

    # Convert data to tensors
    input_size = X_train.shape[1]
    x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Reshape for binary targets
    x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Initialize the model, criterion, and optimizer
    model = SimpleNNBinary(input_size, hidden_size)
    criterion = nn.BCEWithLogitsLoss()  # This applies sigmoid + binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train_tensor)
        loss = criterion(logits, y_train_tensor)
        loss.backward()
        optimizer.step()

    fit_time = time.perf_counter()
    # Evaluation and retrieving logits
    model.eval()
    with torch.no_grad():
        logits = model(x_test_tensor)  # Logits without applying sigmoid
        predictions = torch.sigmoid(logits)  # Apply sigmoid to get probabilities
        predictions = (predictions >= 0.5).float()

    return logits, predictions, fit_time





def run_model(X,Y, method, printer=False):

    start_time = time.perf_counter()
    X_train, X_test, y_train, y_test = X["train"],X["test"],Y["train"],Y["test"]

    """ #SVC ORIGINAL -> Erhöht auf 10x, weil als einziges die Standartabweichung zu hoch ist
    if method == 'svc':
        model = SGDClassifier(loss='hinge', random_state=None)
        model.fit(X_train, y_train)
        fit_time = time.perf_counter()
        y_pred = model.predict(X_test)
        logits = model.decision_function(X_test)
    """
    if method == 'svc': #x10 da zu große Standartabweichung
        logits_accumulative = []
        for i in range(10):
            
            model = SGDClassifier(loss='hinge', random_state=None)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            fit_time = time.perf_counter()
            logits = model.decision_function(X_test)
            logits_accumulative.append(logits)

        
        tmp = np.stack(logits_accumulative)
        logits = np.mean(tmp, axis=0)
        
    elif method == 'randomforest':
        model = RandomForestClassifier(n_estimators=100,n_jobs=-2)
        model.fit(X_train, y_train)
        fit_time = time.perf_counter()

        logits = model.predict_proba(X_test)
        logits = np.max(logits, axis=1)
        y_pred = model.predict(X_test)
        
    elif method == 'logisticregression':
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        fit_time = time.perf_counter()

        logits = model.decision_function(X_test)
        y_pred = model.predict(X_test)
        
    elif method == 'simplenn':
        logits, y_pred,fit_time = run_simplenn_binary(X_train, X_test, y_train, y_test, hidden_size=64, epochs=200)
        
        
    else:
        raise ValueError("Unknown method specified.")
    
    predict_time = time.perf_counter()

    if printer:
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=5, zero_division=1)
        print(f"Accuracy: {accuracy:.4f}")
        print(report)

    data_logits = cov_acc_df_creator(logits,y_test, printer=printer)

    clf_fit_Time_ = fit_time - start_time
    clf_predict_time_ = predict_time - fit_time
    if method == 'svc':
        clf_predict_time_= clf_predict_time_*10
    data_logits["accuracy_coverage"]["fit_time"] = clf_fit_Time_
    data_logits["accuracy_coverage"]["test_time"] = clf_predict_time_
    
    return data_logits

def run_all_models(x, y, printer=False): #svc,randomforest,logisticregression,simplenn
    data_rf = run_model(x, y, method="randomforest",printer=printer)
    data_lr = run_model(x, y, method="logisticregression",printer=printer)
    data_nn = run_model(x, y, method="simplenn",printer=printer)
    data_svc = run_model(x, y, method="svc",printer=printer)
    return {"svc":data_svc,"rf":data_rf,"lr":data_lr,"nn":data_nn}

from transformers import DistilBertModel, DistilBertTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, BertForSequenceClassification



def Bert_embeddings(X_Dataset,modelname, max_length_embedding=128):

    if modelname == "roberta-base":
        complete_name = "roberta-base"
        tokenizer = RobertaTokenizer.from_pretrained(complete_name)
    elif modelname == "distilbert":
        complete_name = "distilbert-base-uncased"
        tokenizer = DistilBertTokenizer.from_pretrained(complete_name)
    elif modelname == "bert":
        complete_name = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(complete_name)
    elif modelname == "tinybert":
        complete_name = "huawei-noah/TinyBERT_General_4L_312D"
        tokenizer = AutoTokenizer.from_pretrained(complete_name,clean_up_tokenization_spaces=False)
    else:
        print("Nicht unterstütztes Modell:", modelname)
        return None, None
    

    model = AutoModel.from_pretrained(complete_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #------------------------------------------------------------------------
    encodings = tokenizer(X_Dataset, truncation=True,
                          padding='max_length', max_length=max_length_embedding, return_tensors='pt') #tokenizer max length like 1 billiarde wtf
    #------------------------------------------------------------------------
    dataset = TensorDataset(
        encodings['input_ids'],        
        encodings['attention_mask'],    
    )
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    cls_embeddings_list = []
    model.eval()
    with torch.no_grad():
        for h,batch in enumerate(tqdm(data_loader)):
            input_ids, attention_mask = [x.to(device) for x in batch]  # Move inputs to device

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            cls_embeddings = cls_embeddings.cpu()
            cls_embeddings_list.append(cls_embeddings)

            del input_ids,attention_mask,outputs,batch

    cls_embeddings_tinyBert = torch.cat(cls_embeddings_list, dim=0).numpy()


    del model, encodings, data_loader,dataset
    torch.cuda.empty_cache()
    gc.collect()    


    return cls_embeddings_tinyBert

def Bert_embeddings_full(X_Dataset,max_length_embedding,modelname):
    start = time.perf_counter()
    embeddings_tinyBert_train = Bert_embeddings(X_Dataset["train"],modelname,max_length_embedding)
    tinyBert_time_train = time.perf_counter() - start
    start = time.perf_counter()
    embeddings_tinyBert_test = Bert_embeddings(X_Dataset["test"],modelname,max_length_embedding)
    tinyBert_time_test = time.perf_counter() - start
    embeddings_tinyBert = {"train":embeddings_tinyBert_train, "test":embeddings_tinyBert_test}
    return embeddings_tinyBert,tinyBert_time_train,tinyBert_time_test


from sklearn.feature_extraction.text import TfidfVectorizer
"""
def tfidf_embeddings(corpus):
    vectorizer = TfidfVectorizer()
    X_train_corpus = vectorizer.fit_transform(corpus)
    X_train_corpus.toarray()
    return X_train_corpus
"""
# Time Versions, 
def tfidf_embeddings(corpus,max_features=None): 

    start_time = time.perf_counter()
    vectorizer = TfidfVectorizer(max_features=max_features) # Max Feature Zahl ist beliebig gewählt
    vectorizer.fit(corpus["train"])
    transformed_corpus_train = vectorizer.transform(corpus["train"]).toarray()
    fit_Time_ = time.perf_counter() - start_time
    
    
    start_time = time.perf_counter()
    transformed_corpus_test = vectorizer.transform(corpus["test"]).toarray()
    test_time_ = time.perf_counter() - start_time

    transformed_corpus = {"train":transformed_corpus_train,"test":transformed_corpus_test}
    #------------------------------------------------
    return transformed_corpus, fit_Time_, test_time_

from sklearn.ensemble import IsolationForest #- Score Sollte outlier sein eigentlich, seltsamer weise +1 manchmal besser
def returnBetterUncertainty_df(scores,match_vector,printer=False): #Manchmal ist -1, oder 1 OutlierScore besser, entscheide was richtig ist nach Performance Score

    df_plus = cov_acc_df_creator(scores,match_vector,printer=printer)
    df_minus = cov_acc_df_creator(scores*-1,match_vector,printer=printer)
    summe_plus,_ = interpolate_and_score(df_plus["accuracy_coverage"]["Coverage"],df_plus["accuracy_coverage"]["Accuracy"])
    summe_minus,_ = interpolate_and_score(df_minus["accuracy_coverage"]["Coverage"],df_minus["accuracy_coverage"]["Accuracy"])
    print("-PScore:",summe_minus, "+PScore:",summe_plus)
    
    if summe_minus >= summe_plus:
        return df_minus
    else:
        return df_plus

from scipy.interpolate import interp1d
from scipy.integrate import simpson
def interpolate_and_score(cov_list,acc_list): #NEUE INTERPOLATIONS VARIANTE
    df_dup = pd.DataFrame({'Coverage': cov_list, 'Accuracy': acc_list})
    df_unique = df_dup.drop_duplicates(subset='Coverage', keep='first')
    x,y = df_unique['Coverage'], df_unique['Accuracy']

    interp_func = interp1d(x,y, kind='linear', fill_value='extrapolate')
    part1 = np.linspace(0, 0.9, int(0.9 / 0.05) + 1)
    part2 = np.linspace(0.91, 1, int((1 - 0.9) / 0.01) + 1)
    new_coverage = np.concatenate((part1, part2)) #Coverage in 1% bis 0.9 dann in 5%
    interpolated_accuracy = interp_func(new_coverage)
    
    #####SCORE BERECHNUNG -> AREA UNDER CURVE APPROX mit simpson (zeiteffizient, non kontinuirliche Funktion...)
    lower_limit = 0.5
    upper_limit = 1.0
    new_coverage_for_integration_x = np.linspace(lower_limit, upper_limit, 1000)
    interpolated_accuracy_for_integration_y = interp_func(new_coverage_for_integration_x)
    score = simpson(interpolated_accuracy_for_integration_y, new_coverage_for_integration_x) # area
    ####

    data_interpolated = {
        'Accuracy': interpolated_accuracy,
        'Coverage': new_coverage,
        #'Score' : [area]len(new_coverage),
    }
    
    return score, data_interpolated

#from sklearn.preprocessing import StandardScaler
#Outlier Detection, Isoforrest
#https://scikit-learn.org/stable/modules/outlier_detection.html
#Berechnet Scores, entscheidet Extra was Outlier sind, erstellt finalen Dataframe

def IsoForest_full(embeddings, match_vector,random_state=None):

    start_time = time.perf_counter()
    iso_forest = IsolationForest(contamination=0.1,n_estimators=300,random_state=random_state, n_jobs=-2)
    iso_forest.fit(embeddings["train"])
    fit_time = time.perf_counter()
    scores = iso_forest.decision_function(embeddings["test"])
    predict_time = time.perf_counter()
    uncertainty_base = returnBetterUncertainty_df(scores,match_vector["test"],printer=False)

    fit_Time_ = fit_time - start_time    #CASE SENSITIVE ACHTUNG
    predict_Time_ = predict_time - fit_time
    uncertainty_base["accuracy_coverage"]["fit_time"] = fit_Time_
    uncertainty_base["accuracy_coverage"]["test_time"] = predict_Time_
    return uncertainty_base
    

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def dummy_data(X,match_vector):

    start_time = time.perf_counter()
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X["train"], match_vector["train"])
    fit_time = time.perf_counter()

    y_pred = model.predict(X["test"])
    predict_time = time.perf_counter()

    report = classification_report(match_vector["test"], y_pred, digits=5, output_dict=True,zero_division=1)
    precision_class_1 = report['True']['precision'] #Problem wenn True Klass nicht Most Frequent
    #precision_class_1 = report['accuracy'] # Da most Frequent, ist Accuracy  nur durch eine Klasse bestimmt, nur Recall 1. bei Most frequent, 0 beim anderen -> accuracy nehmbar direkt
    if int(precision_class_1) == 1: #Falls Most Frequent Class False ist, was passiert wenn Accuracy unter 0.5 ist
        precision_class_1 = 1- report['False']['precision']

    data_logits_dummy = {
        'Accuracy': [precision_class_1]*11,
        'Coverage': [i*0.1 for i in range(11)]
    }

    fit_Time_ = fit_time - start_time    #CASE SENSITIVE ACHTUNG
    predict_Time_ = predict_time - fit_time
    data_logits_dummy["fit_time"] = fit_Time_
    data_logits_dummy["test_time"] = predict_Time_

    pred_score = [int(precision_class_1)] * match_vector["test"]
    return_dict = cov_acc_df_creator(pred_score, match_vector["test"], printer=False)
    return_dict["accuracy_coverage"] = data_logits_dummy #überschreiben wahrscheinlich nicht notwendig, hier aber alle 0.1 X ein Punkt für die Grafik
    
    return return_dict

class StreamToQueue:
    def __init__(self, queue):
        self.queue = queue

    def write(self, message):
        if message != '\n':  # Avoid adding empty newlines
            self.queue.put(message)  # Send the message to the parent process

    def flush(self):
        pass  # No-op to satisfy the "flush" method requirement


from sklearn.metrics import accuracy_score
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from torch import nn
from torch import optim
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

#ACHTUNG fp16 nur mit GPU die es supported
#Trainer Optimisierte Version 
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    BertTokenizer, BertForSequenceClassification,
    AutoTokenizer, AutoModelForSequenceClassification
)
#roberta-base,distilbert,tinybert,bert
def BertRejector(X_Dataset, match_vector, max_length_embedding=128, modelname="roberta-base", result_queue=None,log_queue=None):


    if modelname == "roberta-base":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=1)
    elif modelname == "distilbert":
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)
    elif modelname == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
    elif modelname == "tinybert":
        tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D",clean_up_tokenization_spaces=False)
        model = AutoModelForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", num_labels=1)
    else:
        print("Nicht unterstütztes Modell:", modelname)
        return None, None
    
    if log_queue is not None:  
         stdout_original = sys.stdout
         sys.stdout = StreamToQueue(log_queue) #dict
        #sys.stderr = StreamToQueue(log_queue) #progress bar
        #log_queue.put(f"Initialisiert: {modelname}, seperater Process")
    else:
        print(f"Initialisiert: {modelname}, Main Process")


    # Step 1: Train-test split
    #X_train, X_test, Y_train, Y_test = X_Dataset["train"],X_Dataset["test"],match_vector["train"].astype(float),match_vector["test"].astype(float)

    # Step 2: Prepare the training dataset using Hugging Face Dataset format
    train_data_dict = {"text": X_Dataset["train"], "label": match_vector["train"].astype(float)}
    train_dataset = Dataset.from_dict(train_data_dict)

    # Step 3: Prepare the test dataset
    test_data_dict = {"text": X_Dataset["test"], "label": match_vector["test"].astype(float)}
    test_dataset = Dataset.from_dict(test_data_dict)


    def tokenize_function(examples):
        return tokenizer(examples["text"],
                        padding="max_length", truncation=True, max_length=max_length_embedding) #max_length wäre 512

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)

    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
    # Manually clear the progress bar (datasets uses tqdm internally)
    for bar in tqdm._instances:
        if isinstance(bar, tqdm):  # Check if it's an instance of tqdm
            bar.clear()  # Clear the progress bar
    

    # Step 6: Define training arguments
    training_args = TrainingArguments(
        output_dir="./results", #output_dir="/dev/null" für linux ,Windows: "NUL" wenn kein Folder erstellt werden soll
        eval_strategy="epoch",  
        #eval_steps=500,  # Evaluate every 500 steps
        learning_rate=2e-5,  
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2, 
        weight_decay=0.1,  
        logging_dir=None, 
        logging_steps=10,
        fp16=True, #3060 supported, gibt x2 boost 
        save_strategy="no",
        disable_tqdm=False,
        # gradient_accumulation_steps=2,  # Uncomment if using smaller batch size
    )


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.squeeze(predictions)  
        #mse = mean_squared_error(labels, predictions)
        accuracy = accuracy_score(labels, torch.round(torch.sigmoid(torch.tensor(predictions))).numpy())    
        data_ = cov_acc_df_creator(predictions, labels.astype(bool) ,printer=False) 
        data_ = data_["accuracy_coverage"]

        df_dup = pd.DataFrame({'Coverage': data_['Coverage'], 'Accuracy': data_['Accuracy']})
        df_unique = df_dup.drop_duplicates(subset='Coverage', keep='first')
        interp_func = interp1d(df_unique['Coverage'],df_unique['Accuracy'], kind='linear', fill_value='extrapolate')
        new_coverage_for_integration_x = np.linspace(0.5, 1.0, 100)
        interpolated_accuracy_for_integration_y = interp_func(new_coverage_for_integration_x)
        score = simpson(interpolated_accuracy_for_integration_y, new_coverage_for_integration_x) # area
        
        return {"accuracy": accuracy, "score": score} #validation loss ist actually MSE bei Regression hier

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics, 
    )

    train_output = trainer.train()


    training_time = train_output.metrics['train_runtime']

    predictions = trainer.predict(tokenized_test_dataset)
    logits = np.squeeze(predictions.predictions)
    test_time = predictions.metrics["test_runtime"]

    data_bert_last = cov_acc_df_creator(logits.reshape(-1, 1), match_vector["test"] ,printer=False)
    data_bert_last["accuracy_coverage"]["fit_time"] = training_time
    data_bert_last["accuracy_coverage"]["test_time"] = test_time

    del trainer,model,tokenized_train_dataset,tokenized_test_dataset,tokenizer
    torch.cuda.empty_cache()


    

    if result_queue is not None:    #Eigener Thread
        sys.stdout = stdout_original
        result_queue.put(data_bert_last)
    else:
        return data_bert_last


import multiprocessing as mp
import ast

#GPU Memory bekomm ich nie ganz leer, deswegen seperater Process um GPU Memory voll zu clearen, Macht eigenen TQDM estimator mit drin :)
#Falls das hier Probleme macht, mit default BertRejector ersetzen
#tqdm total=2, da alle BertRejectoren einfach 2 machen
def BertRejector_separate_process(X_Dataset,match_vector,max_length_embedding,modelname):

    result_queue = mp.Queue()
    log_queue = mp.Queue()
    process = mp.Process(target=BertRejector, args=(X_Dataset,match_vector,max_length_embedding,modelname, result_queue,log_queue))
    process.start()
    
    with tqdm(total=2, desc=f"{modelname} Progress", dynamic_ncols=True, ncols=100) as progress_bar:
        while process.is_alive():
            while not log_queue.empty():
                log_message = log_queue.get() 
                if 'epoch' in log_message:
                    try:
                        epoch_value = ast.literal_eval(log_message)["epoch"]
                    except (ValueError, SyntaxError) as e:
                        print("BertRejector_separate_process encountered an error:", e, "Log message:", log_message)
                        epoch_value = 1.9 # Quickfix
                        pass
                    progress_bar.n = epoch_value  
                    progress_bar.last_print_n = epoch_value  

                    progress_bar.update(0) 


            process.join(0)
        
        

    process.join()
    result = result_queue.get()
    progress_bar.close()
    return result


#QA Match Vector/Embedding Generation
from transformers import AutoModelForQuestionAnswering
def QA_MatchVector_Embedding(Data_X,Data_Y,model_name,max_length_embedding=368): #question,context,label(wörter)

    config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name,clean_up_tokenization_spaces=False)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)


    encodings = tokenizer(Data_X, truncation=True,
                        padding='max_length', max_length=max_length_embedding, return_tensors='pt')


    dataset = TensorDataset(
        encodings['input_ids'],        
        encodings['attention_mask'],    
    )


    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    cls_embeddings_list = []
    predicted_answers = []


    model.eval()
    start_time = time.perf_counter()
    with torch.no_grad():
        for h,batch in enumerate(tqdm(data_loader)):
            input_ids, attention_mask = [x.to(device) for x in batch]  # Move inputs to device

            # Forward pass through the model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Get hidden states from the last layer
            hidden_states = outputs.hidden_states

            # Extract CLS token embeddings (first token in the sequence for each input)
            cls_embeddings = hidden_states[-1][:, 0, :].cpu()  
            cls_embeddings_list.append(cls_embeddings)

            # Extract start and end logits from the model's output
            start_logits = outputs.start_logits.cpu()
            end_logits = outputs.end_logits.cpu()

            # Process each example in the batch

            input_ids = input_ids.cpu()
            for i in range(start_logits.size(0)):  # For each example in the batch
                start_index = torch.argmax(start_logits[i]).item()  # Start position of the answer
                end_index = torch.argmax(end_logits[i]).item()  # End position of the answer

                # Convert token indices back to text
                all_tokens = tokenizer.convert_ids_to_tokens(input_ids[i].numpy())  # Convert to tokens
                answer_tokens = all_tokens[start_index:end_index + 1]  # Extract answer tokens
                answer = tokenizer.convert_tokens_to_string(answer_tokens)  # Convert tokens to string
                
                predicted_answers.append(answer.strip())

        del batch,input_ids,attention_mask,outputs,hidden_states,start_logits,end_logits
        torch.cuda.empty_cache()
            

    cls_embeddings_base = torch.cat(cls_embeddings_list, dim=0).numpy()  # Shape: (total_examples, hidden_size)
    predicted_answers = np.array(predicted_answers)
    #answers in filtered_dataset['answers']

    match_vector = predicted_answers == np.array(Data_Y)
    MatchVectorTime = time.perf_counter() - start_time

    #result_dict = {"match_vector": match_vector,"embeddings_base": cls_embeddings_base}
    result_dict = {"match_vector": match_vector,"embeddings_base": cls_embeddings_base,
                   "results":predicted_answers, "softmax":None, "logits":None, "label": None,"MatchVectorTime":MatchVectorTime}
    
    print(np.sum(match_vector),len(match_vector))
    print(np.sum(match_vector)/len(match_vector))


    del model
    torch.cuda.empty_cache()
    gc.collect()

    return result_dict



def CLF_MatchVector_Embedding(X_Dataset,match_vector_labels,model_name,max_length_embedding=128):
    config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
    #tokenizer = AutoTokenizer.from_pretrained(model_name,clean_up_tokenization_spaces=False)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base",clean_up_tokenization_spaces=False)
    #-----------------------------------------------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config) ###!AutoModelForSequenceClassification/AutoModelForQuestionAnswering  usw..
    encodings = tokenizer(X_Dataset, truncation=True, 
                          padding='max_length', max_length=max_length_embedding, return_tensors='pt') #model.config.max_position_embeddings
    #------------------------------------------------------------------------


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = TensorDataset(
        encodings['input_ids'],        # Tokenized input ids
        encodings['attention_mask'],    # Attention masks
    )

    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    cls_embeddings_list = []
    predicted_answers = []
    model.eval()

    start_time = time.perf_counter()
    
    with torch.no_grad():
        for h,batch in enumerate(tqdm(data_loader)):
            input_ids, attention_mask = [x.to(device) for x in batch]  # Move inputs to device

            # Forward pass through the model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings_list.append(outputs.hidden_states[-1][:,0,:])
            predicted_answers.append(outputs.logits.cpu())

            del input_ids, attention_mask
        del batch, data_loader

    embeddings_base = torch.cat(cls_embeddings_list, dim=0).cpu().numpy()  # Shape: (total_examples, hidden_size)
    del cls_embeddings_list
    torch.cuda.empty_cache()

    all_logits = torch.cat(predicted_answers, dim=0)
    scores = softmax(all_logits, axis=-1)
    ranking = np.argsort(scores,axis=-1)
    predicted_labels_list = ranking[:, -1]
    match_vector = predicted_labels_list == np.array(match_vector_labels)

    MatchVectorTime = time.perf_counter() - start_time
    del model,encodings,predicted_answers,dataset
    torch.cuda.empty_cache()
    gc.collect()

    result_dict = {"match_vector": match_vector,"embeddings_base": embeddings_base, "results":predicted_labels_list, "softmax":scores, "logits":all_logits.numpy(), "label": np.array(match_vector_labels),"MatchVectorTime":MatchVectorTime}

    return result_dict



def Regression_MatchVector_Embedding(X_Dataset,match_vector_labels,model_name,max_length_embedding=128,dropout=False,dropout_rate=0.1):
    config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base",clean_up_tokenization_spaces=False)
    #tokenizer = AutoTokenizer.from_pretrained(model_name,clean_up_tokenization_spaces=False)

    #-----------------------------------------------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config) ###!AutoModelForSequenceClassification/AutoModelForQuestionAnswering  usw..
    encodings = tokenizer(X_Dataset, truncation=True, 
                          padding='max_length', max_length=max_length_embedding, return_tensors='pt') #model.config.max_position_embeddings
    #------------------------------------------------------------------------

    if dropout:
        enable_dropout(model,dropout_rate)#default 0.1 dropout

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = TensorDataset(
        encodings['input_ids'],        # Tokenized input ids
        encodings['attention_mask'],    # Attention masks
    )

    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    cls_embeddings_list = []
    predicted_answers = []
    #model.eval()

    start_time = time.perf_counter()
    
    with torch.no_grad():
        for h,batch in enumerate(tqdm(data_loader)):
            input_ids, attention_mask = [x.to(device) for x in batch]  # Move inputs to device

            # Forward pass through the model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings_list.append(outputs.hidden_states[-1][:,0,:])
            predicted_answers.append(outputs.logits.cpu())

            del input_ids, attention_mask
            torch.cuda.empty_cache()
        del batch, data_loader
        torch.cuda.empty_cache()

    embeddings_base = torch.cat(cls_embeddings_list, dim=0).cpu().numpy()  # Shape: (total_examples, hidden_size)
    del cls_embeddings_list
    torch.cuda.empty_cache()

    all_logits = torch.cat(predicted_answers, dim=0)

    if model_name == "cfdd/roberta_timeseries":
        result_logits = all_logits.numpy().reshape(-1)
        differences = np.abs(match_vector_labels - all_logits.numpy().reshape(-1))
        match_vector = differences < 0.3
        results = differences
    elif "cfdd/roberta_regression" in model_name:
        result_logits = all_logits.numpy().reshape(-1)
        y_pred = np.round(all_logits.numpy()).reshape(-1)
        match_vector = y_pred == np.array(match_vector_labels)
        results = y_pred
    else:
        print("Matchvector wird je nach Datensatz individuell definiert in Regression_MatchVector_Embedding, falscher Datensatz angegeben")

    print(np.sum(match_vector),len(match_vector))
    print(np.sum(match_vector)/len(match_vector))

    MatchVectorTime = time.perf_counter() - start_time

    del model,encodings,predicted_answers,dataset
    torch.cuda.empty_cache()
    gc.collect()


    #result_dict = {"match_vector": match_vector,"embeddings_base": embeddings_base, "results":y_pred, "result_logits":all_logits.numpy().reshape(-1), "results_label": np.array(match_vector_labels)}
    result_dict = {"match_vector": match_vector,"embeddings_base": embeddings_base,
                   "results":results, "result_logits":result_logits, "results_label": np.array(match_vector_labels),"MatchVectorTime":MatchVectorTime}


    return result_dict


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
def Transformation_MatchVector_Embedding(X_Dataset,match_vector_labels,model_name,max_length_embedding=64):

    model_name = "oliverguhr/spelling-correction-english-base"
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name,clean_up_tokenization_spaces=False)

    # Load the Seq2Seq model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)

    # Move model to device (GPU if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize the inputs for generation
    encodings = tokenizer(
        X_Dataset, 
        truncation=True, 
        padding='max_length', 
        max_length=max_length_embedding,  # Adjust max_length as needed
        return_tensors='pt'
    )

    # Prepare dataset and dataloader
    dataset = TensorDataset(
        encodings['input_ids'],        # Tokenized input ids
        encodings['attention_mask'],   # Attention masks
    )

    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Lists to store CLS embeddings and predicted corrections
    cls_embeddings_list = []
    decoder_embeddings_list = []
    predicted_answers = []

    # Switch the model to evaluation mode
    model.eval()
    start_time = time.perf_counter()
    # Iterate over the batches
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids, attention_mask = [x.to(device) for x in batch]  # Move inputs to device

            # Forward pass through the model (get encoder's hidden states)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            cls_embeddings_list.append(outputs.encoder_hidden_states[-1][:, 0, :].cpu())

            decoder_embeddings_list.append(outputs.decoder_hidden_states[-1][:, -1, :].cpu())

            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length_embedding)
            predictions = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            predicted_answers.extend(predictions)

            # Free up memory
            del input_ids, attention_mask, generated_ids,outputs
            torch.cuda.empty_cache()


        del batch, data_loader
        torch.cuda.empty_cache()

    embeddings_base = torch.cat(cls_embeddings_list, dim=0).cpu().numpy()  # Shape: (total_examples, hidden_size)
    decoder_base = torch.cat(decoder_embeddings_list, dim=0).cpu().numpy()
    del cls_embeddings_list,decoder_embeddings_list
    torch.cuda.empty_cache()

    match_vector = np.array(predicted_answers) == np.array(match_vector_labels)
    #result_dict = {"match_vector": match_vector,"embeddings_base": decoder_base} #decover embeddings ist hier das relevante
    MatchVectorTime = time.perf_counter() - start_time
    result_dict = {"match_vector": match_vector,"embeddings_base": decoder_base,
                   "results":np.array(predicted_answers), "softmax":None, "logits":None, "label": np.array(match_vector_labels),"MatchVectorTime":MatchVectorTime}    
    return result_dict





import pandas as pd

#Alldata wird nur bei Auswertung benutzt
def create_pandas_frame(DataDict,Dataset_name,Method_name,Data_size,CLF_name,alldata=False):
    if alldata: 
        df = pd.DataFrame(DataDict)
    else:
        df = pd.DataFrame({key: DataDict["accuracy_coverage"][key] for key in ['Accuracy', 'Coverage']}) # Accuracy', 'Coverage', 'fit_time', 'test_time
    df['Dataset'] = Dataset_name
    df['Method'] = Method_name
    df['Data_size'] = Data_size
    df['Classifier'] = CLF_name
    if alldata:
        return df


    #All in one Pandas bei sovielen Zusatzdaten langsam schlecht, workaround für Datasize ist extra Dict für ROC
    Method_Classifier = Method_name + ' ' + CLF_name
    roc_dict ={
        "Method_Classifier":Method_Classifier,
        "Data_size":Data_size,
        "roc_auc": DataDict["roc_auc"],
        "FPR":DataDict["FPR"],
        "TPR":DataDict["TPR"]
        
    }
    return df,roc_dict

from scipy.stats import entropy

def entropy_softmax_calc(softmax,match_vector): #Benötigt kein Fit, braucht eigentlich nur Test set, daher nur Testset berechnet
    start_time = time.perf_counter()
    entropy_softmax = {split: np.apply_along_axis(entropy, axis=1, arr=softmax[split]) for split in ["train", "test"]}
    all_Time_ = time.perf_counter() - start_time
    
    entropy_softmax_df = cov_acc_df_creator(entropy_softmax["test"]*-1,match_vector["test"])
    entropy_softmax_df["accuracy_coverage"]["fit_time"] = 0
    ratio = len(entropy_softmax["test"])/(len(entropy_softmax["train"])+len(entropy_softmax["test"]))
    entropy_softmax_df["accuracy_coverage"]["test_time"] = all_Time_*ratio #2/12 ist Ratio 

    return entropy_softmax_df



def softmax_score_calc(softmax,match_vector): #Benötigt kein Fit, braucht eigentlich nur Test set, daher nur Testset berechnet
    start_time = time.perf_counter()
    softmax_score = {split: np.max(softmax[split], axis=1).reshape(-1) for split in ["train", "test"]}
    all_Time_ = time.perf_counter() - start_time
    
    softmax_score_df = cov_acc_df_creator(softmax_score["test"],match_vector["test"])
    softmax_score_df["accuracy_coverage"]["fit_time"] = 0
    ratio = len(softmax_score["test"])/(len(softmax_score["train"])+len(softmax_score["test"]))
    softmax_score_df["accuracy_coverage"]["test_time"] = all_Time_*ratio #2/12 ist Ratio 

    return softmax_score_df



#def enable_dropout(model):
#    for layer in model.modules():
#        if isinstance(layer, nn.Dropout):
#            layer.train()  # Enable dropout during evaluation


def enable_dropout(model, p=None):
    for layer in model.modules():
        if isinstance(layer, nn.Dropout):
            layer.train()  # Enable dropout
            if p is not None:
                layer.p = p  # Update dropout probability


#CLF_Embedding Clone mit Dropout
#activate_mc_dropout results bei random= 0.0 ~ gleiche varianz wie bei random= 0.1 (huggingface standart dropout)
#0.2 macht ~doppelte varianz, 0.3 dann dreifache im vergleich 0.0 bzw 0.1, sollte eigentlich bei 0.0 kein Dropout sein, aber negligible + gehört probably dazu
#0.3 dropout hier weil https://arxiv.org/pdf/2205.03109 auch hat für uncertainty Estimation
def CLF_Dropout(X_Dataset,match_vector_labels,model_name,max_length_embedding=128,dropout_runs = 10,dropout_rate=0.1):

    start_time = time.perf_counter()
    config = AutoConfig.from_pretrained(model_name, output_hidden_states=True, hidden_dropout_prob=dropout_rate, attention_probs_dropout_prob=dropout_rate)
    #tokenizer = AutoTokenizer.from_pretrained(model_name,clean_up_tokenization_spaces=False)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base",clean_up_tokenization_spaces=False)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config) 
    enable_dropout(model) #default dropout ist 0.1, muss man nicht einstellen 

    encodings = tokenizer(X_Dataset, truncation=True, 
                          padding='max_length', max_length=max_length_embedding, return_tensors='pt') #model.config.max_position_embeddings
    
    dataset = TensorDataset(
        encodings['input_ids'],        
        encodings['attention_mask'],    
    )
  
    
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ModelLoadTime = time.perf_counter() - start_time

    dropout_dicts = []
    for run in range(dropout_runs):

        data_loader = DataLoader(dataset, batch_size=16, shuffle=False)
        cls_embeddings_list = []
        predicted_answers = []
        with torch.no_grad():
            for h,batch in enumerate(tqdm(data_loader)):
                input_ids, attention_mask = [x.to(device) for x in batch]  # Move inputs to device

                # Forward pass through the model
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                cls_embeddings_list.append(outputs.hidden_states[-1][:,0,:])
                predicted_answers.append(outputs.logits.cpu())

                del input_ids, attention_mask
            del batch, data_loader
            torch.cuda.empty_cache()

        embeddings_base = torch.cat(cls_embeddings_list, dim=0).cpu().numpy()  # Shape: (total_examples, hidden_size)
        del cls_embeddings_list
        torch.cuda.empty_cache()

        all_logits = torch.cat(predicted_answers, dim=0)
        scores = softmax(all_logits, axis=-1)
        ranking = np.argsort(scores,axis=-1)
        predicted_labels_list = ranking[:, -1]
        match_vector = predicted_labels_list == np.array(match_vector_labels)

        result_dict = {
            "match_vector": match_vector,
            "embeddings_base": embeddings_base,  
            "results": predicted_labels_list,
            "softmax": scores,
            "logits": all_logits.numpy(),
            "label": np.array(match_vector_labels),

        }
        
        dropout_dicts.append(result_dict)

    del model,encodings,predicted_answers,dataset
    torch.cuda.empty_cache()
    gc.collect()
    
    return dropout_dicts,ModelLoadTime


def CLF_Dropout_FULL(X_Dataset,match_vector_labels,model_name,max_length_embedding,match_vector,
                     dropout_runs,dropout_rate):

    start_time = time.perf_counter()
    dropout_dicts,ModelLoadTime = CLF_Dropout(X_Dataset,match_vector_labels,model_name,max_length_embedding=max_length_embedding,
                                dropout_runs=dropout_runs,dropout_rate=dropout_rate)

    #dict_keys(['match_vector', 'embeddings_base', 'results', 'softmax', 'logits', 'label'])
    all_results = np.array([dropout_dicts[run]["results"] for run in range(dropout_runs)])
    all_logits = np.array([dropout_dicts[run]["logits"] for run in range(dropout_runs)])
    all_softmax = np.array([dropout_dicts[run]["softmax"] for run in range(dropout_runs)]) 


    from scipy import stats
    dropout_result,dropout_match_count = stats.mode(all_results, axis=0, keepdims=False)
    dropout_match_vector = dropout_result == dropout_dicts[0]["label"]

    #majority/agreement score
    agreement_score = dropout_match_count/dropout_runs

    softmax_ensemble_vals = np.transpose(all_softmax, (1, 0, 2))
    softmax_max_ = np.max(softmax_ensemble_vals,axis=2)
    softmax_max_mean = np.mean(softmax_max_,axis=1)

    #ab hier df creation 
    end_time = time.perf_counter()
    mcdropout_time = end_time - start_time - ModelLoadTime #Zeit zum Laden Modell wird abgezogen

    result_softmax_max_mean = cov_acc_df_creator(softmax_max_mean,match_vector["test"])
    result_softmax_max_mean_dp = cov_acc_df_creator(softmax_max_mean,dropout_match_vector)

    agreement_dropout = cov_acc_df_creator(agreement_score,dropout_match_vector)
    agreement_match_vector = cov_acc_df_creator(agreement_score,match_vector["test"])

    return_dict = {"softmax_mv":result_softmax_max_mean,"softmax_dp":result_softmax_max_mean_dp,"agreement_mv":agreement_match_vector,"agreement_dp":agreement_dropout}

    for method,data in return_dict.items():
        data["accuracy_coverage"]["test_time"] = mcdropout_time
    
    return return_dict



from collections import Counter
# Define a function to compute the most common value and its frequency
def most_common_and_frequency(arr, axis=1):
    def mode_func(x):
        counter = Counter(x)
        most_common_value, frequency = counter.most_common(1)[0]
        return most_common_value, frequency
    
    # Apply along the specified axis
    results = np.apply_along_axis(lambda x: mode_func(x), axis=axis, arr=arr)
    most_common_values = np.array([r[0] for r in results])
    frequencies = np.array([r[1] for r in results])
    return most_common_values, frequencies.astype(int)


def QA_Dropout(Data_X,Data_Y,model_name,max_length_embedding=368,dropout_rate=0.1): 

    config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name,clean_up_tokenization_spaces=False)

    enable_dropout(model,p=dropout_rate)
    
    # Move model to device (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize the dataset (questions + contexts)
    encodings = tokenizer(Data_X, truncation=True,
                        padding='max_length', max_length=max_length_embedding, return_tensors='pt')

    # Create a TensorDataset for the DataLoader
    dataset = TensorDataset(
        encodings['input_ids'],        # Tokenized input ids
        encodings['attention_mask'],    # Attention masks
    )

    # Create DataLoader for batching

    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    cls_embeddings_list = []
    predicted_answers = []

    # Disable gradient calculations for inference
    start_time = time.perf_counter()
    with torch.no_grad():
        for h,batch in enumerate(tqdm(data_loader)):
            input_ids, attention_mask = [x.to(device) for x in batch]  # Move inputs to device

            # Forward pass through the model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Get hidden states from the last layer
            hidden_states = outputs.hidden_states

            # Extract CLS token embeddings (first token in the sequence for each input)
            cls_embeddings = hidden_states[-1][:, 0, :].cpu()  # Shape: (batch_size, hidden_size)
            cls_embeddings_list.append(cls_embeddings)

            # Extract start and end logits from the model's output
            start_logits = outputs.start_logits.cpu()
            end_logits = outputs.end_logits.cpu()

            # Process each example in the batch

            input_ids = input_ids.cpu()
            for i in range(start_logits.size(0)):  # For each example in the batch
                start_index = torch.argmax(start_logits[i]).item()  # Start position of the answer
                end_index = torch.argmax(end_logits[i]).item()  # End position of the answer

                # Convert token indices back to text
                all_tokens = tokenizer.convert_ids_to_tokens(input_ids[i].numpy())  # Convert to tokens
                answer_tokens = all_tokens[start_index:end_index + 1]  # Extract answer tokens
                answer = tokenizer.convert_tokens_to_string(answer_tokens)  # Convert tokens to string
                
                predicted_answers.append(answer.strip())

        del batch,input_ids,attention_mask,outputs,hidden_states,start_logits,end_logits
        torch.cuda.empty_cache()
            

    cls_embeddings_base = torch.cat(cls_embeddings_list, dim=0).numpy()  # Shape: (total_examples, hidden_size)
    predicted_answers = np.array(predicted_answers)
    #answers in filtered_dataset['answers']

    match_vector = predicted_answers == np.array(Data_Y)
    MatchVectorTime = time.perf_counter() - start_time

    result_dict = {"results":predicted_answers,"MatchVectorTime":MatchVectorTime}#"CLS":cls_embeddings_base

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return result_dict


#Spellcheck Dropout

def Transformation_Dropout(X_Dataset,match_vector_labels,model_name,max_length_embedding=64,dropout_rate=0.1):

    model_name = "oliverguhr/spelling-correction-english-base"
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name,clean_up_tokenization_spaces=False)

    # Load the Seq2Seq model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
    enable_dropout(model,p=dropout_rate)
    # Move model to device (GPU if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize the inputs for generation
    encodings = tokenizer(
        X_Dataset, 
        truncation=True, 
        padding='max_length', 
        max_length=max_length_embedding,  # Adjust max_length as needed
        return_tensors='pt'
    )

    # Prepare dataset and dataloader
    dataset = TensorDataset(
        encodings['input_ids'],        # Tokenized input ids
        encodings['attention_mask'],   # Attention masks
    )

    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Lists to store CLS embeddings and predicted corrections
    cls_embeddings_list = []
    decoder_embeddings_list = []
    predicted_answers = []

    # Switch the model to evaluation mode
    start_time = time.perf_counter()
    # Iterate over the batches
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids, attention_mask = [x.to(device) for x in batch]  # Move inputs to device

            # Forward pass through the model (get encoder's hidden states)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            cls_embeddings_list.append(outputs.encoder_hidden_states[-1][:, 0, :].cpu())

            decoder_embeddings_list.append(outputs.decoder_hidden_states[-1][:, -1, :].cpu())

            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length_embedding)
            predictions = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            predicted_answers.extend(predictions)

            # Free up memory
            del input_ids, attention_mask, generated_ids,outputs
            torch.cuda.empty_cache()


        del batch, data_loader
        torch.cuda.empty_cache()

    embeddings_base = torch.cat(cls_embeddings_list, dim=0).cpu().numpy()  # Shape: (total_examples, hidden_size)
    decoder_base = torch.cat(decoder_embeddings_list, dim=0).cpu().numpy()
    del cls_embeddings_list,decoder_embeddings_list
    torch.cuda.empty_cache()

    match_vector = np.array(predicted_answers) == np.array(match_vector_labels)
    #result_dict = {"match_vector": match_vector,"embeddings_base": decoder_base} #decover embeddings ist hier das relevante
    MatchVectorTime = time.perf_counter() - start_time
    result_dict = {"results":np.array(predicted_answers),"MatchVectorTime":MatchVectorTime}    
    return result_dict








#def QA_dropout_Full(Data_X,Data_Y,model_name,max_length_embedding=368,match_vector,match_vector_labels):
def Frequency_dropout_Full(X_Dataset,match_vector_labels,model_name,max_length_embedding,match_vector,
                     dropout_runs,dropout_rate,origin_dataset):
    return_dict = {}

    if origin_dataset == "Question Answering":
        dropout_inference = QA_Dropout
        print("QA Dropout Start")
    elif origin_dataset == "Transformation":
        dropout_inference = Transformation_Dropout
        print("Transformation Spellcheck Dropout Start")
    else:
        print("Dropout Fail in Frequency Dropout")
        raise ValueError('Dropout Error')
    
    dropout_results = []
    for i in range(dropout_runs):
        arr = dropout_inference(X_Dataset["test"],match_vector_labels["test"],model_name,max_length_embedding,dropout_rate)
        dropout_results.append(arr)


    time_val_dropout = sum([arr["MatchVectorTime"] for arr in dropout_results])
    dropout_results_array = np.array([arr["results"] for arr in dropout_results]).T
    most_common_values, frequencies = most_common_and_frequency(dropout_results_array, axis=1)



    #mv_test === ACTUAL MATCH VECTOR match_vector["test"]
    #written_label = match vector labels test
    #match_vector_labels = {"train": labels_train,"test" : labels_test}

    mv_test = match_vector["test"]
    written_label = np.array(match_vector_labels["test"]) 
    mv_val_dropout = most_common_values == written_label[:len(most_common_values)]
    
    agreement_mv = cov_acc_df_creator(frequencies/dropout_runs,mv_test)
    agreement_dp = cov_acc_df_creator(frequencies/dropout_runs,mv_val_dropout)
    agreement_mv["accuracy_coverage"]["test_time"] = time_val_dropout
    agreement_dp["accuracy_coverage"]["test_time"] = time_val_dropout
    return_dict["agreement_mv"] = agreement_mv
    return_dict["agreement_dp"] = agreement_dp
    #written_label = np.array([label_dict[y] for y in datasets[Dataset_name]["train"]["label"]])
    print(np.sum(mv_val_dropout)/len(mv_val_dropout))
    return return_dict


def timeseries_dropout_cleaner(dropout_results):
    
    predictions = np.array([arr["result_logits"] for arr in dropout_results]).T
    
    average_clean = []
    var_clean = []
    for predictions in predictions:
        average_clean.append(np.average(np.array(predictions)))
        var_clean.append(np.var(np.array(predictions)))
    return np.array(average_clean),np.array(var_clean)
    
def Variance_dropout_Full(X_Dataset,match_vector_labels,model_name,max_length_embedding,match_vector,
                     dropout_runs,dropout_rate,Dataset_name):
    return_dict = {}

    
    dropout_results = []
    for i in range(dropout_runs):
        dropout_return_dict = Regression_MatchVector_Embedding(X_Dataset["test"],match_vector_labels["test"],
                                                 model_name,max_length_embedding=128,dropout=True,dropout_rate=0.1)
        
        dropout_results.append(dropout_return_dict)


    time_val_dropout = sum([arr["MatchVectorTime"] for arr in dropout_results])
    average, var = timeseries_dropout_cleaner(dropout_results)
    
    
    mv_test = match_vector["test"]
    test_label = np.array(match_vector_labels["test"])
    
    if Dataset_name == "Time Series Regression":
        differences = np.abs(test_label[:len(average)] - average)
        mv_val_dropout = differences < 0.3
    elif Dataset_name == "Regression":
        preds = np.round(average)
        mv_val_dropout = preds == test_label
    else:
        raise ValueError("Datasetmixxup in Variance Dropout")
    
    agreement_mv = cov_acc_df_creator(var*-1,mv_test)
    agreement_dp = cov_acc_df_creator(var*-1,mv_val_dropout)
    agreement_mv["accuracy_coverage"]["test_time"] = time_val_dropout
    agreement_dp["accuracy_coverage"]["test_time"] = time_val_dropout
    return_dict["agreement_mv"] = agreement_mv
    return_dict["agreement_dp"] = agreement_dp
    #written_label = np.array([label_dict[y] for y in datasets[Dataset_name]["train"]["label"]])
    print("Dropout Result",np.sum(mv_val_dropout)/len(mv_val_dropout))
    return return_dict



#Bad Practice blabla...
def timer(Classification_Dict,match_vector_time,results_embedding,
          Outlier_Dict,data_logits_dummy,bert_rejectors):
    
    TimeDict = {}
    existing_embeddings = []
    for model in results_embedding.keys():
        existing_embeddings.append(f"{model} Embedding")
        TimeDict[f"{model} Embedding"] = {}
    TimeDict["Tf-Idf Embedding"] = {}

    if Classification_Dict !=None:
        print("Info: Available Classification_Dict") 

        if set(Classification_Dict.keys()) == {'dropout_agreement_mv', 'dropout_agreement_dp'}: #dropout methoden ohne softmax 
            print("Info: Only Dropout Precalucalted, no Softmax") 
        else:
            print("Info: Full Classification_Dict Available") 
            #BEIDE KEINE INITIAL KOSTEN - Brauchen Basis Modelrun(=Match_vector_time) NUR FÜR TEST -- Berechnen Softmax & Entropy
            TimeDict["Softmax"] = {"Initial": {"MatchVectorTime": 0, "fit_time": 0},
                                          "Inference":{"Model Inference":match_vector_time["test"], "test_time": Classification_Dict["softmax"]["accuracy_coverage"]["test_time"]}}
            TimeDict["Dropout Softmax MV"] = {"Initial": {"MatchVectorTime": 0, "fit_time": 0},
                                          "Inference":{"Model Inference": 0, "test_time": Classification_Dict["dropout_softmax_mv"]["accuracy_coverage"]["test_time"]}}
            TimeDict["Dropout Softmax DP"] = {"Initial": {"MatchVectorTime": 0, "fit_time": 0},
                                          "Inference":{"Model Inference": 0, "test_time": Classification_Dict["dropout_softmax_dp"]["accuracy_coverage"]["test_time"]}}
        #Gleiche Bei Dropout, keine Initial Kosten
        TimeDict["Dropout Agreement MV"] = {"Initial": {"MatchVectorTime": 0, "fit_time": 0},
                                      "Inference":{"Model Inference": 0, "test_time": Classification_Dict["dropout_agreement_mv"]["accuracy_coverage"]["test_time"]}}
        TimeDict["Dropout Agreement DP"] = {"Initial": {"MatchVectorTime": 0, "fit_time": 0},
                                      "Inference":{"Model Inference": 0, "test_time": Classification_Dict["dropout_agreement_dp"]["accuracy_coverage"]["test_time"]}}

    else:
        print("Info: No Classification_Dict Available") 

    for classifier in ['svc', 'rf', 'lr', 'nn']:
        #Alle Embeddings Types DistilBert,TinyBert.. X alle Classifier NN,LR.. Types
        for model, model_data in results_embedding.items():
            if model == "Base":
                TimeDict[f"{model} Embedding"][classifier] = {"Initial": {"MatchVectorTime":match_vector_time["train"], "fit_time": model_data[classifier]["accuracy_coverage"]["fit_time"]},
                                      "Inference":{"Model Inference":match_vector_time["test"], "test_time": model_data[classifier]["accuracy_coverage"]["test_time"]}}
            else:
                TimeDict[f"{model} Embedding"][classifier] = {"Initial": {"MatchVectorTime":match_vector_time["train"],
                                                                    "Method Time":model_data["Method_Fit_Time"],
                                                                    "fit_time": model_data[classifier]["accuracy_coverage"]["fit_time"]},
                                                        "Inference":{"Method Time":model_data["Method_Test_Time"],
                                                                    "test_time": model_data[classifier]["accuracy_coverage"]["test_time"]}}

        #TimeDict["Tf-Idf Embedding"][classifier] = {"Initial": {"MatchVectorTime":match_vector_time["train"],
        #                                                     "Method Time":data_logits_tfidf_dict["Method_Fit_Time"],
        #                                                     "fit_time": data_logits_tfidf_dict[classifier]["fit_time"]},
        #                                        "Inference":{"Method Time":data_logits_tfidf_dict["Method_Test_Time"],
        #                                                     "test_time": data_logits_tfidf_dict[classifier]["test_time"]}}
    
    #data_logits_tiny_dict=results_embedding[name]
    #Outlier Zeit Berechnung-> Inference: Test Embedding Creation Method time(Results_Embedding) + Isoforrest Test time(Outlier Dict)

    for model, outlier_data in Outlier_Dict.items():
        if model == "Base":
            TimeDict["Base Outlier"] = {"Initial":{"MatchVectorTime":match_vector_time["train"],"fit_time": outlier_data["accuracy_coverage"]["fit_time"]},
                               "Inference":{"MatchVectorTime":match_vector_time["test"],"test_time": outlier_data["accuracy_coverage"]["test_time"]}}
        else:
            name = f"{model} Outlier"
            TimeDict[name] = {"Initial":{"MatchVectorTime":match_vector_time["train"],"Method Time":results_embedding[model]["Method_Fit_Time"],"fit_time": outlier_data["accuracy_coverage"]["fit_time"]},
                                "Inference":{"Method Time":results_embedding[model]["Method_Test_Time"],"test_time": outlier_data["accuracy_coverage"]["test_time"]}} 


    #TimeDict["Tfidf Outlier"] = {"Initial":{"MatchVectorTime":match_vector_time["train"],"Method Time":data_logits_tfidf_dict["Method_Fit_Time"],"fit_time": uncertainty_tfidf["fit_time"]},
    #                           "Inference":{"Method Time":data_logits_tfidf_dict["Method_Test_Time"],"test_time": uncertainty_tfidf["test_time"]}} #Muss Tinybert methode kopieren, ist richtig

    TimeDict["Most Frequent Class"] = {"Initial":{"MatchVectorTime":match_vector_time["train"],"fit_time": data_logits_dummy["accuracy_coverage"]["fit_time"]},
                                      "Inference":{"test_time": 0}} #Input Egal, immer True -> Kein Tokenizer notwendig oder andere Zeit
    #BertRejector Zeit Berechnung
    for rejector, rejector_data in bert_rejectors.items():
        name = rejector + " Rejector"
        TimeDict[name] = {"Initial":{"MatchVectorTime":match_vector_time["train"],"fit_time":rejector_data["accuracy_coverage"]["fit_time"]},"Inference":{"test_time":rejector_data["accuracy_coverage"]["test_time"]}}

    #Baut aus TimeDict neue Format, was ich in Pandas umwandeln kann, unnötig kompliziert sorry
    #----------------
    #VizList format - MethodenName,List_InitalTime,List_InferTime

    nameDict = {'svc':"SVM", 'rf':"Random Forest", 'lr':"Logistic Regression", 'nn':"Simple NN"}
    VizList = []
    multi_classifier_methods = existing_embeddings #extra loop für Methoden mit mehreren Classifier z.B ["Base Embeddings","Tf-Idf Embedding"] 
    for method in multi_classifier_methods:
        for classifier in ['svc', 'rf', 'lr', 'nn']:
            name = method + " " + nameDict[classifier]
            inital = []
            for key in TimeDict[method][classifier]["Initial"].keys():
                inital.append((key,TimeDict[method][classifier]["Initial"][key]))

            infer = []
            for key in TimeDict[method][classifier]["Inference"].keys():
                infer.append((key,TimeDict[method][classifier]["Inference"][key]))

            VizList.append([name,inital,infer])

    for method in TimeDict.keys()-multi_classifier_methods:
        name = method + " generic"
        inital = []
        for key in TimeDict[method]["Initial"].keys():
            inital.append((key,TimeDict[method]["Initial"][key]))

        infer = []
        for key in TimeDict[method]["Inference"].keys():
            infer.append((key,TimeDict[method]["Inference"][key]))

        VizList.append([name,inital,infer])

    #Pandas Format  
    rows = []
    for method, init_times, infer_times in VizList:
        init_dict = dict(init_times)
        infer_dict = dict(infer_times)
        rows.append({
            'Method': method,
            'Initial Time': sum(init_dict.values()),
            'Inference Time': sum(infer_dict.values()),
            'Initial Info': init_dict,
            'Inference Info': infer_dict
        })

    df_time = pd.DataFrame(rows)
    return df_time  


def unify_dicts(dict_list):
    unified = {}
    
    for data in dict_list:
        # Extract values
        method_classifier = data['Method_Classifier']
        data_size = data['Data_size']
        metrics = {key: data[key] for key in ['roc_auc', 'FPR', 'TPR']}
        
        # Insert into unified dict
        if data_size not in unified:
            unified[data_size] = {}
        if method_classifier not in unified[data_size]:
            unified[data_size][method_classifier] = {}
        
        # Update metrics
        unified[data_size][method_classifier] = metrics
    
    return unified