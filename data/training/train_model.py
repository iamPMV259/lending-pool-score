import asyncio

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from clients import Clients
from configs import get_logger
from mongo.schemas import PoolsSnapshotTest, PoolsSnapshotTrain

logger = get_logger(__name__)
mongo_client = Clients.get_mongo_client()


FEATURE_COLUMNS = [
    'tvl_current', 
    'tvl_mean', 
    'tvl_volatility', 
    'max_drawdown', 
    'apy_mean', 
    'apy_std', 
    'chain_score'
]


TARGET_COLUMN = 'label'


async def load_data_from_mongo(collection_class):
    logger.info(f"Loading data from {collection_class.__name__} collection.")

    records = await collection_class.find_many().to_list()

    if not records:
        logger.warning(f"No records found in {collection_class.__name__} collection.")
        return pd.DataFrame()

    data = [r.model_dump() for r in records]
    df = pd.DataFrame(data)
    return df


async def train_and_evaluate():
    await mongo_client.initialize()

    df_train = await load_data_from_mongo(PoolsSnapshotTrain)
    df_test = await load_data_from_mongo(PoolsSnapshotTest)


    if df_train.empty or df_test.empty:
        logger.error("Training or testing data is empty. Aborting training process.")
        return

    logger.info(f"Data Loaded: Train shape: {df_train.shape}, Test shape: {df_test.shape}")

    X_train = df_train[FEATURE_COLUMNS]
    y_train = df_train[TARGET_COLUMN]
    X_test = df_test[FEATURE_COLUMNS]
    y_test = df_test[TARGET_COLUMN]


    logger.info("Start Training Random Forest...")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    logger.info("Model training completed.")
    logger.info("Evaluating model on test data...")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\n" + "="*40)
    print(f"Accuracy: {acc * 100:.2f}%")
    print("="*40)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    model_filename = "random_forest_model.pkl"
    joblib.dump(model, model_filename)
    logger.info(f"Trained model saved to {model_filename}.")


if __name__ == "__main__":
    asyncio.run(train_and_evaluate())