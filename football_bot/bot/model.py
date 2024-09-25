from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.pipeline import make_pipeline

def train_model(features, labels):
    # Check if features and labels are empty
    if features.empty or labels.empty:
        raise ValueError("Features or labels are empty.")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define models
    xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42)
    lgb_model = lgb.LGBMClassifier(random_state=42)

    # Create an ensemble model using VotingClassifier
    ensemble_model = VotingClassifier(
        estimators=[
            ('xgboost', xgb_model),
            ('lightgbm', lgb_model)
        ],
        voting='soft'
    )

    # Perform cross-validation
    cv_scores = cross_val_score(ensemble_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

    # Train the ensemble model
    ensemble_model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = ensemble_model.score(X_test, y_test)
    y_pred = ensemble_model.predict(X_test)
    report = classification_report(y_test, y_pred)

    return ensemble_model, accuracy, report
