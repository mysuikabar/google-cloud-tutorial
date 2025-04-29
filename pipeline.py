import joblib
import kfp
import pandas as pd
from kfp import dsl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


@dsl.component
def load_data() -> str:
    # サンプルデータの作成
    data = {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [5, 4, 3, 2, 1],
        "target": [0, 1, 0, 1, 0],
    }
    df = pd.DataFrame(data)
    df.to_csv("data.csv", index=False)
    return "data.csv"


@dsl.component
def preprocess_data(data_path: str) -> str:
    df = pd.read_csv(data_path)
    # 簡単な前処理
    X = df[["feature1", "feature2"]]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # データの保存
    X_train.to_csv("X_train.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)

    return "X_train.csv"


@dsl.component
def train_model(X_train_path: str, y_train_path: str) -> str:
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    model = RandomForestClassifier()
    model.fit(X_train, y_train.values.ravel())

    # モデルの保存
    model_path = "model.joblib"
    joblib.dump(model, model_path)
    return model_path


@dsl.pipeline(
    name="ml-pipeline",
    description="A simple ML pipeline with data preprocessing and model training",
)
def ml_pipeline() -> None:
    # データの読み込み
    load_data_task = load_data()

    # データの前処理
    preprocess_data(data_path=load_data_task.output)

    # モデルのトレーニング
    train_model(X_train_path="X_train.csv", y_train_path="y_train.csv")


if __name__ == "__main__":
    # パイプラインをコンパイル
    kfp.compiler.Compiler().compile(
        pipeline_func=ml_pipeline, package_path="ml_pipeline.yaml"
    )
