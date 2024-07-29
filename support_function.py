import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, log_loss, roc_curve
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import torch
import torch.nn as nn
import torch.optim as optim


class DefaultProbModel(nn.Module):
    def __init__(self, input_size, layers):
        super(DefaultProbModel, self).__init__()

        # List to hold all the layers
        nn_layers = []

        # Initialize current size to the input size
        current_size = input_size
        
        # Iterate through the layer sizes to create the layers
        for size in layers:

            # Check is normal layer or not
            if size >= 1:

                # Add a Linear layer
                nn_layers.append(nn.Linear(current_size, size))

                # Update current size to the new layer's size
                current_size = size

                # If not the last layer, add ReLU activation
                if size != layers[-1]:
                    nn_layers.append(nn.ReLU())
            
            # # If not the last layer, add ReLU activation
            # if size != layers[-1]:
            #     nn_layers.append(nn.ReLU())

            # If size < 1, then add a Dropout with the probability
            if size < 1:
                nn_layers.append(nn.Dropout(size))
            
            # If it is the last layer, add Sigmoid activation
            if size == layers[-1]:
                nn_layers.append(nn.Sigmoid())

        # Use nn.Sequential to combine all the layers into one model
        self.model = nn.Sequential(*nn_layers)

    def forward(self, x):
        # Forward pass: pass the input through the model
        return self.model(x)

    
def train_nn_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=1500, eval_interval=500):
   # Set the model to training mode
   model.train()
   for epoch in range(num_epochs):
        
        # Initialize the total loss for this epoch
        running_loss = 0
        
        # Training loop
        for X_train, y_train in train_loader:
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(X_train)
            # Compute the loss
            loss = criterion(outputs, y_train)
            # Backward pass
            loss.backward()
            # Update the weights
            optimizer.step()
          
            running_loss += loss.item()
        
        # Evaluate the model every `eval_interval` epochs
        if (epoch + 1) % eval_interval == 0:

            # Set the model to evaluation mode
            model.eval()

            # Initialize the total loss for this epoch
            val_loss = 0
            
            # Training loop
            for X_val, y_val in val_loader:
                # Zero the gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = model(X_val)
                # Compute the loss
                loss = criterion(outputs, y_val)
                # Backward pass
                loss.backward()
                # Update the weights
                optimizer.step()
                
                val_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}')


def prepare_data_for_nn(X_train, X_val, y_train, y_val):
    # Scale the features using StandardScaler to standardize the features
    scaler = StandardScaler()
    
    # Fit the scaler on the training data and transform it
    X_train_std = scaler.fit_transform(X_train)
    
    # Transform the validation data using the same scaler
    X_val_std = scaler.transform(X_val)

    # Convert the standardized training and validation data to PyTorch tensors with dtype float32
    X_train_tensor = torch.tensor(X_train_std, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_std, dtype=torch.float32)
    
    # Convert the training and validation labels to PyTorch tensors with dtype float32 and reshape to column vector
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1)
    y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32).view(-1, 1)

    # Create data loaders for training and validation data
    train_loader = [(X_train_tensor, y_train_tensor)]
    val_loader = [(X_val_tensor, y_val_tensor)]

    # Return the data loaders
    return train_loader, val_loader


def train_val_test(
        df: pd.DataFrame,
        train_val_test_ratio=[0.7, 0.15, 0.15]) -> pd.DataFrame:
    """
    This function splits the dataframe into training, validation, and test sets based on the specified ratio.
    
    Parameters:
        df (pd.DataFrame): Dataframe
        train_val_test_ratio (list): List of three float values representing the ratio for training, validation, and test sets. Default is [0.7, 0.15, 0.15].
    
    Returns:
        tuple: A tuple containing the feature and target DataFrames for training, validation, and test sets:
               (X_train, X_val, X_test, y_train, y_val, y_test)
               or None if the required target column is not found.
    """
    
    # Shuffle the DataFrame using NumPy
    np.random.seed(21)
    shuffled_indices = np.random.permutation(len(df))
    df = df.iloc[shuffled_indices].reset_index(drop=True)
    
    # Calculate split indices
    train_end = int(train_val_test_ratio[0] * len(df))
    val_end = train_end + int(train_val_test_ratio[1] * len(df))
    
    # Split the DataFrame into train, validation, and test sets
    train = df[:train_end].reset_index(drop=True)
    val = df[train_end:val_end].reset_index(drop=True)
    test = df[val_end:].reset_index(drop=True)
    
    # Check for the presence of the 'target' column and split accordingly
    if 'target' in df.columns:
        X_train = train.drop(['target'], axis=1)
        y_train = train['target']
        X_val = val.drop(['target'], axis=1)
        y_val = val['target']
        X_test = test.drop(['target'], axis=1)
        y_test = test['target']
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    # Return None if 'target' columns are found
    else:
        return None
    

def print_metrics(y_true, y_pred):
    """
    Calculate and print various classification metrics: accuracy, precision, precision, F1, ROC AUC.

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.
    """

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, )
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'F1: {f1:.2f}')
    print(f'ROC AUC: {roc_auc:.2f}')


class ModelTrainer:
    def __init__(self, model_type, version, model_save_path, model_name_note):
        self.model_type = model_type
        self.version = int(version)
        self.model_save_path = model_save_path
        self.model_name_note = model_name_note


    def print_title(self, title, length=120):
        formatted_title = f' {title} '.center(length, '=')
        print('\n' + formatted_title + '\n')


    def standardize_data(self, X_train, X_val):
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_val_std = scaler.transform(X_val)
        return X_train_std, X_val_std


    def print_metrics(self, y_pred, y_true):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)

        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'F1: {f1:.2f}')
        print(f'ROC AUC: {roc_auc:.2f}')

        return [accuracy, precision, f1, roc_auc]
    

    def train_and_evaluate(self, X_train, y_train, X_val, y_val, opt_trials, layer_sizes=[1], num_epoch=2000, lr=1e-3, save=True, **hyperparameters):

        if self.model_type == 'logistics':
            self.print_title('Logistics Regression')
            X_train, X_val = self.standardize_data(X_train, X_val)

        elif self.model_type == 'gb':
            self.print_title('Gradient Boosting Classification')

        elif self.model_type == 'svm':
            self.print_title('Support Vector Machine')
            X_train, X_val = self.standardize_data(X_train, X_val)

        elif self.model_type == 'nb':
            self.print_title('Naive Bayes')
            X_train, X_val = self.standardize_data(X_train, X_val)

        elif self.model_type == 'nn':
            self.print_title('Neural Network')
            train_loader, val_loader = prepare_data_for_nn(X_train, X_val, y_train, y_val)
        else:
            raise ValueError("Unsupported model type. Choose from 'logistics', 'gb', 'svm', 'nb' or 'nn'.")

        # Hyperparameter Tuning is on
        if opt_trials > 0:

            def objective(params):

                if self.model_type in ['logistics', 'gb', 'svm', 'nb']:

                    if self.model_type == 'logistics':
                        model = LogisticRegression(C=params['C'], random_state=21)

                    if self.model_type == 'nb':
                        model = BernoulliNB(alpha=params['alpha'])

                    elif self.model_type == 'gb':
                        model = GradientBoostingClassifier(
                            n_estimators=int(params['n_estimators']),
                            learning_rate=params['learning_rate'],
                            max_depth=int(params['max_depth']),
                            min_samples_split=int(params['min_samples_split']),
                            min_samples_leaf=int(params['min_samples_leaf']),
                            subsample=params['subsample'],
                            max_features=params['max_features'],
                            random_state=21
                        )

                    elif self.model_type == 'svm':
                        model = SVC(
                            C=params['C'],
                            kernel=params['kernel'],
                            degree=int(params['degree']),
                            gamma=params['gamma']
                        )

                    model.fit(X_train, y_train)
                    val_prediction = model.predict(X_val)
                    score = log_loss(y_val, val_prediction)
                    return {'loss': score, 'status': STATUS_OK}

                else:
                    learning_rate = params['learning_rate']
                    weight_decay = params['weight_decay']

                    model = DefaultProbModel(input_size=X_train.shape[1], layers=layer_sizes)
                    criterion = nn.BCELoss()
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

                    epochs = 1000
                    for _ in range(epochs):
                        model.train()
                        for data, target in train_loader:
                            optimizer.zero_grad()
                            output = model(data)
                            loss = criterion(output, target)
                            loss.backward()
                            optimizer.step()

                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for data, target in val_loader:
                            output = model(data)
                            loss = criterion(output, target)
                            val_loss += loss.item()
                    score = val_loss
                    return {'loss': score, 'status': STATUS_OK}

            if self.model_type == 'logistics':
                search_space = {
                    'C': hp.loguniform('C', np.log(1e-3), np.log(1e3))
                }
            
            elif self.model_type == 'nb':
                search_space = {
                    'alpha': hp.loguniform('alpha', np.log(1e-3), np.log(1)),
                    'binarize': hp.uniform('binarize', 0.0, 1.0)
                }
            
            elif self.model_type == 'gb':
                search_space = {
                   'n_estimators': hp.quniform('n_estimators', 50, 500, 10), 
                   'learning_rate': hp.loguniform('learning_rate', np.log(1e-3), np.log(1e-1)), 
                   'max_depth': hp.quniform('max_depth', 3, 15, 1), 
                   'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1), 
                   'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 20, 1), 
                   'subsample': hp.uniform('subsample', 0.0, 1.0), 
                   'max_features': hp.choice('max_features', ['sqrt', 'log2'])
                }
            
            elif self.model_type == 'svm':
                search_space = {
                    'C': hp.loguniform('C', np.log(1e-3), np.log(1e3)),
                    'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                    'degree': hp.quniform('degree', 2, 5, 1),
                    'gamma': hp.choice('gamma', ['scale', 'auto'])
                }
            
            else:
                search_space = {
                    'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-1)),
                    'weight_decay': hp.loguniform('weight_decay', np.log(1e-5), np.log(1e-1))
                }

            trials = Trials()
            best_params = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=opt_trials, trials=trials)
            print(f'Best trial: {best_params}')

            if self.model_type == 'gb':
                max_features_list = ['sqrt', 'log2']
                model = GradientBoostingClassifier(
                    n_estimators=int(best_params['n_estimators']),
                    learning_rate=best_params['learning_rate'],
                    max_depth=int(best_params['max_depth']),
                    min_samples_split=int(best_params['min_samples_split']),
                    min_samples_leaf=int(best_params['min_samples_leaf']),
                    subsample=best_params['subsample'],
                    max_features=max_features_list[best_params['max_features']],
                    random_state=21
                )
            
            elif self.model_type == 'svm':
                kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
                gamma_list = ['scale', 'auto']
                model = SVC(
                    C=best_params['C'],
                    kernel=kernel_list[best_params['kernel']],
                    degree=int(best_params['degree']),
                    gamma=gamma_list[best_params['gamma']]
                )
            
            elif self.model_type == 'logistics':
                model = LogisticRegression(C=best_params['C'], random_state=21)
            
            elif self.model_type == 'nb':
                model = BernoulliNB(
                    alpha=best_params['alpha'], 
                    binarize=best_params['binarize']
                )
            
            else:
                model = DefaultProbModel(input_size=X_train.shape[1], layers=layer_sizes)
                criterion = nn.BCELoss()
                optimizer = optim.Adam(
                    model.parameters(), 
                    lr=best_params['learning_rate'], 
                    weight_decay=best_params['weight_decay']
                )

        else:
            if self.model_type == 'gb':
                model = GradientBoostingClassifier(**hyperparameters)

            elif self.model_type == 'svm':
                model = SVC(**hyperparameters)
            
            elif self.model_type == 'logistics':
                model = LogisticRegression(**hyperparameters)
            
            elif self.model_type == 'nb':
                model = BernoulliNB(**hyperparameters)

            else:
                model = DefaultProbModel(input_size=X_train.shape[1], layers=layer_sizes)
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)

        if self.model_type in ['logistics', 'gb', 'svm', 'nb']:
            model.fit(X_train, y_train)

        else:
            train_nn_model(
                model=model, 
                criterion=criterion, 
                optimizer=optimizer, 
                train_loader=train_loader, 
                val_loader=val_loader, 
                num_epochs=num_epoch
            )

        if save:
            if self.model_name_note is not None:
                final_save_path = f'{self.model_save_path}/{self.model_type}_{self.model_name_note}_v{self.version}'
            else:
                final_save_path = f'{self.model_save_path}/{self.model_type}_v{self.version}'

            if self.model_type == 'nn':
                torch.save(model.state_dict(), final_save_path + '.pth')
            else:
                with open(final_save_path + '.pkl', 'wb') as f:
                    pickle.dump(model, f)

        if self.model_type == 'nn':
            val_prediction = model(val_loader[0][0]).detach().numpy()
            fpr, tpr, thresholds = roc_curve(y_val, val_prediction)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            val_prediction = np.where(val_prediction > optimal_threshold, 1, 0)

        else:
            val_prediction = model.predict(X_val)

        errors = self.print_metrics(val_prediction, y_val)
        
        return model, errors
