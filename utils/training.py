from sklearn.svm import SVC

def train_model(X_train, y_train):

    model = SVC(
        kernel='rbf',
        probability=True,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model