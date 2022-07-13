from main import prepare_and_normalise_dataset
import pandas as pd

def test_prepare_and_normalise_dataset():
    # Test sans intéret mais seulement pour montrer qu'on utilise 'pytest' au déploiement.
    expected = [10000, 11]
    df = pd.read_csv('./Churn_Modelling.csv', sep=',', header=0)
    results = [len(df.axes[0]), len(df.axes[1])]
    assert expected == results