import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

Datasets_list = ['Africa',
                 'BankNote',
                 'CreditApproval',
                 'CreditRisk',
                 'CreditCard',
                 'Ecoli',
                 'FakeBills',
                 'LoanPrediction',
                 'PageBlock',
                 'PageBlockDel',
                 'PredictTerm',
                 'SouthGermanCredit',
                 'Wine',
                 'WineRed',
                 'Yeast',
                 'YeastUn']


def print_datasets_list():
    print(Datasets_list)


class Dataset:
    def __init__(self, dataset_name):
        self.DatasetName = dataset_name
        self.Loader = None

    def GetDataset(self):
        if self.DatasetName not in Datasets_list:
            print('Err.')
        else:
            if self.DatasetName == 'Africa':
                return self._load_Africa()
            elif self.DatasetName == 'BankNote':
                return self._load_BankNote()
            elif self.DatasetName == 'CreditApproval':
                return self._load_CreditApproval()
            elif self.DatasetName == 'CreditRisk':
                return self._load_CreditRisk()
            elif self.DatasetName == 'CreditCard':
                return self._load_CreditCard()
            elif self.DatasetName == 'Ecoli':
                return self._load_Ecoli()
            elif self.DatasetName == 'FakeBills':
                return self._load_FakeBills()
            elif self.DatasetName == 'LoanPrediction':
                return self._load_LoanPrediction()
            elif self.DatasetName == 'PageBlock':
                return self._load_PageBlock()
            elif self.DatasetName == 'PageBlockDel':
                return self._load_PageBlockDel()
            elif self.DatasetName == 'PredictTerm':
                return self._load_PredictTerm()
            elif self.DatasetName == 'SouthGermanCredit':
                return self._load_SouthGermanCredit()
            elif self.DatasetName == 'Wine':
                return self._load_Wine()
            elif self.DatasetName == 'WineRed':
                return self._load_WineRed()
            elif self.DatasetName == 'Yeast':
                return self._load_Yeast()
            elif self.DatasetName == 'YeastUn':
                return self._load_YeastUn()

    @staticmethod
    def _load_Africa():
        dataset = pd.read_csv("Datasets/Africa Economic, Banking and Systemic Crisis Data/african_crises.csv")

        X = dataset.drop(['banking_crisis', 'country'], axis=1)
        y = dataset['banking_crisis']
        y = y.replace({'no_crisis': 0, 'crisis': 1})
        y.astype('int')

        le = LabelEncoder()
        X['cc3'] = le.fit_transform(X['cc3'])
        X = MinMaxScaler().fit_transform(X)

        return X, y

    @staticmethod
    def _load_BankNote():
        dataset = pd.read_csv("Datasets/Bank Note Authentication UCI data/BankNote_Authentication.csv")

        X = dataset.drop(['class'], axis=1)
        y = dataset['class']
        y.astype('int')

        X = MinMaxScaler().fit_transform(X)

        return X, y

    @staticmethod
    def _load_CreditApproval():
        cols = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'class']
        dataset = pd.read_csv('Datasets/Credit Approval/CreditApprovalDataSet.data', names=cols, delimiter=',')
        dataset = dataset.replace('?', np.nan)
        dataset = dataset.dropna()
        X = dataset.drop(['class'], axis=1)
        y = dataset['class']
        y = y.replace({'-': 0, '+': 1})
        y.astype('int')

        le = LabelEncoder()
        X['A1'] = le.fit_transform(X['A1'])
        X['A4'] = le.fit_transform(X['A4'])
        X['A5'] = le.fit_transform(X['A5'])
        X['A6'] = le.fit_transform(X['A6'])
        X['A7'] = le.fit_transform(X['A7'])
        X['A9'] = le.fit_transform(X['A9'])
        X['A10'] = le.fit_transform(X['A10'])
        X['A12'] = le.fit_transform(X['A12'])
        X['A13'] = le.fit_transform(X['A13'])

        X['A2'] = X['A2'].astype(float)
        X['A14'] = X['A14'].astype(float)
        X = MinMaxScaler().fit_transform(X)
        return X, y

    @staticmethod
    def _load_CreditRisk():
        dataset = pd.read_csv("Datasets/Credit Risk/customer_data_drop_na.csv", delimiter=',')
        X = dataset.drop(['label', 'id'], axis=1)
        y = dataset['label']
        y.astype('int')
        X = MinMaxScaler().fit_transform(X)
        return X, y

    @staticmethod
    def _load_CreditCard():
        # dataset = pd.read_excel("datasets/default_of_credit_card_clients.xls", header=1)
        dataset = pd.read_csv("Datasets/Default of Credit Card Clients Dataset/UCI_Credit_Card.csv", delimiter=',')
        X = dataset.drop(['default.payment.next.month', 'ID'], axis=1)
        y = dataset['default.payment.next.month']
        X = MinMaxScaler().fit_transform(X)
        return X, y

    @staticmethod
    def _load_Ecoli():
        dataset = pd.read_csv('Datasets/Ecoli4/ecoli4_new.csv',
                              names=['mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class'],
                              delimiter=',')

        X = dataset.drop(['class'], axis=1)
        y = dataset['class']
        y = y.replace({'negative': 0, 'positive': 1})
        X = MinMaxScaler().fit_transform(X)
        return X, y

    @staticmethod
    def _load_FakeBills():
        # Column names
        cols = ['Class', 'Diagonal', 'Height_left', 'Height_right', 'Margin_low', 'Margin_up', 'Length']
        dataset = pd.read_csv("Datasets/Fake Bills/fake_bills_drop_na.csv", header=None, names=cols)
        dataset = dataset.replace({True: 0, False: 1})

        X = dataset.drop(['Class'], axis=1)
        y = dataset['Class'].astype("int")

        # Scale the datasets
        X = MinMaxScaler().fit_transform(X)

        return X, y

    @staticmethod
    def _load_LoanPrediction():
        dataset = pd.read_csv("Datasets/Loan Prediction Problem Dataset/train_dropna.csv", delimiter=',')
        X = dataset.drop(['Loan_ID', 'Loan_Status'], axis=1)
        y = dataset['Loan_Status']
        y = y.replace({'N': 1, 'Y': 0})
        y.astype('int')
        le = LabelEncoder()
        X['Gender'] = le.fit_transform(X['Gender'])
        X['Married'] = le.fit_transform(X['Married'])
        X['Dependents'] = le.fit_transform(X['Dependents'])
        X['Education'] = le.fit_transform(X['Education'])
        X['Self_Employed'] = le.fit_transform(X['Self_Employed'])
        X['Property_Area'] = le.fit_transform(X['Property_Area'])
        X = MinMaxScaler().fit_transform(X)

        return X, y

    @staticmethod
    def _load_PageBlock():
        dataset = pd.read_csv('Datasets/Page Blocks/page-block_new.csv', delimiter=',')
        X = dataset.drop(['class'], axis=1)
        y = dataset['class']
        y = y.replace({3: 1, 1: 0, 2: 0, 4: 0, 5: 0})
        y.astype('int')

        X = MinMaxScaler().fit_transform(X)

        return X, y

    @staticmethod
    def _load_PageBlockDel():
        dataset = pd.read_csv('Datasets/Page Blocks/page-block_new.csv', delimiter=',')
        dataset = dataset.loc[~((dataset["class"] == 1) | (dataset["class"] == 4)), :].copy()
        X = dataset.drop(['class'], axis=1)
        y = dataset['class']
        y = y.replace({3: 1, 2: 0, 5: 0})
        y.astype('int')

        X = MinMaxScaler().fit_transform(X)

        return X, y

    @staticmethod
    def _load_PredictTerm():
        dataset = pd.read_csv('Datasets/Predict Term Deposit/Assignment-2_Data_dropna.csv', delimiter=',')
        X = dataset.drop(['Id', 'y'], axis=1)
        y = dataset['y']
        y = y.replace({'no': 0, 'yes': 1})
        y.astype('int')
        le = LabelEncoder()
        X['job'] = le.fit_transform(X['job'])
        X['marital'] = le.fit_transform(X['marital'])
        X['education'] = le.fit_transform(X['education'])
        X['default'] = le.fit_transform(X['default'])
        X['housing'] = le.fit_transform(X['housing'])
        X['loan'] = le.fit_transform(X['loan'])
        X['contact'] = le.fit_transform(X['contact'])
        X['month'] = le.fit_transform(X['month'])
        X['poutcome'] = le.fit_transform(X['poutcome'])

        X = MinMaxScaler().fit_transform(X)

        return X, y

    @staticmethod
    def _load_SouthGermanCredit():
        dataset = pd.read_csv("Datasets/South German Credit/SouthGermanCredit.asc", sep=' ')

        X = dataset.drop(['kredit'], axis=1)
        y = dataset['kredit'].astype("int")
        y = y.replace({1: 0, 0: 1})

        X = MinMaxScaler().fit_transform(X)

        return X, y

    @staticmethod
    def _load_Wine():
        cols = ['class', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13']
        dataset = pd.read_csv('Datasets/Wine/wine.data', names=cols, delimiter=',')
        X = dataset.drop(['class'], axis=1)
        y = dataset['class']
        y = y.replace({1: 1, 2: 0, 3: 0})
        y.astype('int')

        X = MinMaxScaler().fit_transform(X)

        return X, y

    @staticmethod
    def _load_WineRed():
        cols = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'class']
        dataset = pd.read_csv("Datasets/Wine/winequality-red-8_vs_6.dat", names=cols, delimiter=',')
        X = dataset.drop(["class"], axis=1)
        y = dataset['class']
        y = y.replace({'negative': 0, 'positive': 1})
        y.astype('int')

        X = MinMaxScaler().fit_transform(X)

        return X, y

    @staticmethod
    def _load_Yeast():
        dataset = pd.read_csv("Datasets/Yeast/yeast_new.csv", delimiter=',')
        dataset = dataset.loc[((dataset["class"] == "ME2") | (dataset["class"] == "CYT"))]
        X = dataset.drop(["Sequence Name", "class"], axis=1)
        y = dataset['class']
        y = y.replace({"ME2": 1, "CYT": 0})
        y.astype('int')

        X = MinMaxScaler().fit_transform(X)

        return X, y

    @staticmethod
    def _load_YeastUn():
        y_map = {"ME2": 1, "CYT": 0, "NUC": 0, "MIT": 0, "ME1": 0, "ME3": 0, "EXC": 0, "VAC": 0, "POX": 0, "ERL": 0}
        dataset = pd.read_csv("Datasets/Yeast/yeast_new.csv", delimiter=',')
        X = dataset.drop(["Sequence Name", "class"], axis=1)
        y = dataset['class']
        y = y.replace(y_map)
        y.astype('int')

        X = MinMaxScaler().fit_transform(X)

        return X, y


def Check(dataset_name):
    dataset = Dataset(dataset_name=dataset_name)
    X, y = dataset.GetDataset()
    minor, major = 0, 0
    for i in y:
        if i == 1:
            minor += 1
        else:
            major += 1
    print(f"minor: {minor}, major: {major}")
    assert minor < major


def Full_check():
    for i in Datasets_list:
        print(f"Dataset {i}:")
        Check(i)
        print('\n')


if __name__ == "__main__":
    Full_check()
