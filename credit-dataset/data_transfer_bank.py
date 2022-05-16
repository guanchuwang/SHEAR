import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

import pickle

class PandasDataSet(TensorDataset):

    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame('dummy')
        return torch.from_numpy(df.values).float()


def load_ICU_data(path):
    column_names = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y']
    input_data = pd.read_csv(path) # , names=column_names)
    # input_data = (pd.read_csv(path, names=column_names,
    #                           na_values="?", sep=r'\s*,\s*', engine='python'
    #                           ).loc[lambda df: df['race'].isin(['White', 'Black'])])

    # input_data = (pd.read_csv(path, names=column_names,
    #                           na_values="?", sep=r'\s*,\s*', engine='python'))
    # not_white_index = np.where((input_data['race'] == 'White').values == False)[0]
    # input_data['race'].iloc[not_white_index] = 'not-White'

    # sensitive attributes; we identify 'race' and 'sex' as sensitive attributes
    sensitive_attribs = ['age']
    sensitive_value = input_data.loc[:, sensitive_attribs].values
    Z = pd.DataFrame((np.array([[int(x)] for x in sensitive_value]) > 35).astype(np.int), columns=["age"]) # pd.DataFrame([int(x) for x in sensitive_value], column_names="age")
    # print(Z)
    # input_data.loc[1:, sensitive_attribs] = sensitive_value
    # Z = (input_data.loc[:, sensitive_attribs].assign(age=lambda df: int(df['age']) > 25))

    y = (input_data['y'] == "yes").astype(int)

    categorical_attribs = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
    categorical_value = input_data.loc[:, categorical_attribs].pipe(pd.get_dummies, drop_first=False)
    real_attribs = ["day", "duration", "campaign", "pdays", "previous"] # "balance",
    real_value = input_data.loc[:, real_attribs]

    X = pd.concat([real_value, categorical_value], axis=1)
    # print(input_data)
    # print(real_value)
    # print(categorical_value)
    # print(X)

    # features; note that the 'target' and sentive attribute columns are dropped
    # X = (input_data.drop(columns=["age", "y"]))
         # .fillna('Unknown')
         # .pipe(pd.get_dummies, drop_first=True))

    print(f"features X: {X.shape[0]} samples, {X.shape[1]} attributes")
    print(f"targets y: {y.shape} samples")
    print(f"sensitives Z: {Z.shape[0]} samples, {Z.shape[1]} attributes")
    return X, y, Z

def main():

    # load ICU data set
    X, y, Z = load_ICU_data('./bank_us_bigset.csv') #

    n_instance = X.shape[0]
    n_features = X.shape[1]
    n_sensitive = Z.shape[1]

    print('n_instance:', n_instance)

    # split into train/test set
    # (X_train, X_test, y_train, y_test, Z_train, Z_test) = train_test_split(X, y, Z,
    #                                                                        train_size=30000,
    #                                                                        test_size=900,
    #                                                                        stratify=y, random_state=7)
    #
    # print('Z train Sex minority:', ((Z_train['sex'] == 0)).sum() * 1. / len(Z_train['sex']))
    # print('Z test Sex minority:', ((Z_test['sex'] == 0)).sum()*1./len(Z_test['sex']))
    # print('Y train minority:', ((y_train == 1)).sum()*1./len(y_train))
    # print('Y test minority:', ((y_test == 1)).sum()*1./len(y_test))

    dataset = pd.concat([X, y, Z], axis=1)

    print(dataset.keys())
    print(dataset.values[0, :])

    dataset.to_csv("./bank_bigset_preprocessed.csv")

    # return

    # scaler = StandardScaler().fit(X_train)
    # scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df),
    #                                            columns=df.columns, index=df.index)
    # X_train = X_train.pipe(scale_df, scaler)
    # X_test = X_test.pipe(scale_df, scaler)
    #
    # train_data = PandasDataSet(X_train, y_train, Z_train)
    # test_data = PandasDataSet(X_test, y_test, Z_test)


    # X_train, y_train, Z_train = X_train.values, y_train.values, Z_train.values.squeeze(axis=1)
    # X_test, y_test, Z_test = X_test.values, y_test.values, Z_test.values.squeeze(axis=1)
    #
    # with open('data/adult-data/adult_train.pkl', 'wb') as fileObject:
    #     pickle.dump((X_train, y_train, Z_train), fileObject)  # 保存list到文件
    #     fileObject.close()
    #
    # with open('data/adult-data/adult_test.pkl', 'wb') as fileObject:
    #     pickle.dump((X_test, y_test, Z_test), fileObject)  # 保存list到文件
    #     fileObject.close()
    #
    # with open('data/adult-data/adult_train.pkl', 'rb') as fileObject:
    #     X_train, y_train, Z_train = pickle.load(fileObject)
    #     fileObject.close()
    #
    # with open('data/adult-data/adult_test.pkl', 'rb') as fileObject:
    #     X_test, y_test, Z_test = pickle.load(fileObject)
    #     fileObject.close()
    #
    # print(X_train, y_train, Z_train)
    # print(X_test, y_test, Z_test)

    # np.save('data/adult-data/adult_train.pkl', [X_train, y_train, Z_train])
    # np.save('data/adult-data/adult_test.pkl', [X_test, y_test, Z_test])

    

if __name__ == "__main__":

    torch.manual_seed(7)
    np.random.seed(4) # 11
    # random.seed(7)

    main()