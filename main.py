import pandas as pd
import numpy as np
import warnings
import time

np.set_printoptions(threshold=10000,suppress=True)
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from fastapi import FastAPI
import joblib

app = FastAPI()
clfs = {
        'DT': DecisionTreeClassifier(criterion='entropy', random_state=0),
        'RF': RandomForestClassifier(n_estimators=100, random_state=0),
        'ADA': AdaBoostClassifier(n_estimators=100, random_state=0),
        'MLP': MLPClassifier(hidden_layer_sizes=(20, 10), alpha=0.001, max_iter=200),
    }

run_results = []

@app.post("/evaluate")
def evaluate_classifier():
    Xtrain, Xtest, Ytrain, Ytest = prepare_and_normalise_dataset()

    #print("------------------ Running the ML script ------------------")

    return run_classifier(Xtrain, Ytrain, Xtest, Ytest)

@app.get("/predictions/{data}")
def predict(data: str, classifier: str):

    X = [np.array(data.split(","))]
    try:
        model = joblib.load("models/" + classifier + "_loan_granting.joblib")
        res = model.predict(X)
        print("Prediction : ", res, ". For model ", classifier)

    except ValueError:
        print("Failed to load or predict bc => ", ValueError)

    return res.tolist()


def prepare_and_normalise_dataset():
    df = pd.read_csv('./houses.csv', sep=',', header=0)
    total_rows = len(df.axes[0])
    total_cols = len(df.axes[1])
    print(df.describe())

    #### 1. Séparation du jeu de données (Xtrain) et test (Xtest):

    df.drop(['orientation'], axis=1, inplace=True)
    X = df.iloc[:, :3].values  # = toutes les colonnes sans la dernière (Exited)
    Y = df.iloc[:, 3].values  # = seulement la dernière colonne
    Y=Y.astype(int)

    newY = []
    for i in range(0, len(Y)):
        if Y[i] >= 1234:
            newY.append(0)
        else:
            newY.append(1)

    Y[Y>=279981] = 1
    Y[Y<279981] = 0
    Y=Y.astype(int)
    print(X)
    print(Y)
    print(type(Y[0]))
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25, random_state=1)

    #### 2. Normalisation du jeu de données :

    SS = StandardScaler()
    SS.fit(Xtrain)

    XNormTrain = SS.transform(Xtrain)
    XNormTest = SS.transform(Xtest)


    return XNormTrain, XNormTest, Ytrain, Ytest

def run_classifier(Xtrain, Ytrain, Xtest, Ytest):
    tot1 = time.time()
    run_results = []

    for i in clfs:  # Pour chaque classifieur stocké dans la liste clfs
        clf = clfs[i]
        debut = time.time()

        clf.fit(Xtrain, Ytrain)
        Ypred = clf.predict(Xtest)

        acc = accuracy_score(Ytest, Ypred)
        recall = recall_score(Ytest, Ypred)
        mean_score = (acc + recall) / 2

        print_acc = "Accuracy for {0} is : {1:.3f} % +/- {2:.3f}".format(i, np.mean(acc) * 100, np.std(acc))
        print_recall = "Recall for {0} is : {1:.3f} % +/- {2:.3f}".format(i, np.mean(recall) * 100, np.std(recall))
        print_result = "RESULT ====> : {1:.3f} % +/- {2:.3f}".format(i, np.mean(mean_score) * 100, np.std(mean_score))

        print(print_acc)
        print(print_recall)
        print(print_result)
        run_results.append(print_acc)
        run_results.append(print_recall)
        run_results.append(print_result)

        fin = time.time()
        processingTime = fin - debut

        print_duration = "     Execution time for {0} is : {1:.3f} sec".format(i, processingTime)
        print(print_duration)
        run_results.append(print_duration)
        run_results.append("==============================================")

        modelTitle = "models/" + str(i) + "_loan_granting.joblib"
        joblib.dump(clf, modelTitle)
        print("Model has been saved : ", modelTitle)
        print("------------------------------------\n")

    tot2 = time.time()
    totalProcessingTime = tot2 - tot1
    print(' \n TEMPS TOTAL D EXECUTION : {0:.4f} sec.'.format(totalProcessingTime))
    run_results.append('TEMPS TOTAL D EXECUTION : {0:.4f} sec.'.format(totalProcessingTime))
    return run_results

if __name__ == '__main__':

    Xtrain, Xtest, Ytrain, Ytest = prepare_and_normalise_dataset()

    print("Running the ML script.")

    run_classifier(Xtrain, Ytrain, Xtest, Ytest)

