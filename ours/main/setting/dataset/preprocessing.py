import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


def process_cen_income(data_path):
    col_names = ['age',
                'workclass',
                'fnlwgt',
                'education',
                'education-num',
                'marital-status',
                'occupation',
                'relationship',
                'race',
                'sex',
                'capital-gain',
                'capital-loss',
                'hours-per-week',
                'native-country',
                'income']
    train_data = pd.read_csv(os.path.join(data_path, "adult.data"), sep=',\s', na_values=["?"], engine='python', header=None, names=col_names)
    test_data = pd.read_csv(os.path.join(data_path, "adult.test"), sep=',\s', na_values=["?"], engine='python', header=None, names=col_names)
    test_data = test_data.drop(index=0).reset_index(drop=True)  # delete '|1x3 Cross validator'

    data = pd.concat([train_data, test_data], axis=0)

    missing_columns = data.columns[data.isnull().any()]
    for col in missing_columns:
        data[col].fillna(data[col].mode()[0], inplace=True)
    
    continuous_cols = ['age', 'fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
    category_cols = [col for col in col_names if col not in continuous_cols]

    for col in category_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    sca = MinMaxScaler()
    data[continuous_cols] = sca.fit_transform(data[continuous_cols])
    data[continuous_cols] = data[continuous_cols].round(4)

    labels = data['income']
    data = data.drop(columns=['income'])

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_loader, test_loader

def process_ger_credit(data_path):
    col_names = ["account_status",
               "period",
               "history_credit",
               "credit_purpose",
               "credit_limit",
               "saving_account",
               "person_employee",
               "income_installment_rate",
               "marry_sex",
               "other_debtor",
               "address","property",
               "age",
               "installment_plans",
               "housing",
               "credits_num",
               "job",
               "dependents",
               "have_phone",
               "foreign_worker",
               "target"]
    data = pd.read_csv(os.path.join(data_path, "german.data"), sep='\s', engine='python', header=None, names=col_names)
    data.target = data.target - 1

    continuous_cols = list(data.select_dtypes(include=['int','float','int32','float32','int64','float64']).columns.values)
    category_cols = [x for x in col_names if x not in continuous_cols]
    
    for col in category_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    sca = MinMaxScaler()
    data[continuous_cols] = sca.fit_transform(data[continuous_cols])
    data[continuous_cols] = data[continuous_cols].round(4)

    labels = data['target']
    data = data.drop(columns=['target'])

    print(data.head())

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_loader, test_loader


def process_ban_market(data_path):
    data = pd.read_csv(os.path.join(data_path, "bank-additional-full.csv"), sep=';', na_values=["?"], engine='python', header=0)

    missing_columns = data.columns[data.isnull().any()]
    for col in missing_columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

    print(data.head())

    continuous_cols = list(data.select_dtypes(include=['int','float','int32','float32','int64','float64']).columns.values)
    category_cols = [x for x in data.columns.to_list() if x not in continuous_cols]

    for col in category_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    sca = MinMaxScaler()
    data[continuous_cols] = sca.fit_transform(data[continuous_cols])
    data[continuous_cols] = data[continuous_cols].round(4)
    
    labels = data['y']
    data = data.drop(columns=['y'])

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_loader, test_loader

if __name__ == "__main__":
    process_cen_income('/data/cxx/QuantBackdoor_EFRAP/ours/main/data/census_income')
    process_ger_credit('/data/cxx/QuantBackdoor_EFRAP/ours/main/data/german_credit')
    process_ban_market('/data/cxx/QuantBackdoor_EFRAP/ours/main/data/bank_marketing/bank-additional')