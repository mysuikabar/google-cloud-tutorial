from kfp import compiler
from kfp.dsl import Dataset, Input, Metrics, Output, component, pipeline


@component(packages_to_install=["pandas", "scikit-learn"])
def load_data(output_data: Output[Dataset]) -> None:
    import pandas as pd

    # サンプルデータの作成
    data = {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [5, 4, 3, 2, 1],
        "target": [0, 1, 0, 1, 0],
    }
    df = pd.DataFrame(data)
    df.to_csv(output_data.path, index=False)


@component(packages_to_install=["pandas", "scikit-learn"])
def preprocess_data(
    input_data: Input[Dataset],
    X_train: Output[Dataset],
    y_train: Output[Dataset],
    X_test: Output[Dataset],
    y_test: Output[Dataset],
) -> None:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_data.path)
    # 簡単な前処理
    X = df[["feature1", "feature2"]]
    y = df["target"]
    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X, y, test_size=0.2)

    # データの保存
    X_train_df.to_csv(X_train.path, index=False)
    X_test_df.to_csv(X_test.path, index=False)
    y_train_df.to_csv(y_train.path, index=False)
    y_test_df.to_csv(y_test.path, index=False)


@component(packages_to_install=["pandas", "scikit-learn", "joblib"])
def train_model(
    X_train: Input[Dataset], y_train: Input[Dataset], model: Output[Dataset]
) -> None:
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    X_train_df = pd.read_csv(X_train.path)
    y_train_df = pd.read_csv(y_train.path)

    rf_model = RandomForestClassifier()
    rf_model.fit(X_train_df, y_train_df.values.ravel())

    # モデルの保存
    joblib.dump(rf_model, model.path)


@component(packages_to_install=["pandas", "scikit-learn", "joblib"])
def evaluate_model(
    X_test: Input[Dataset],
    y_test: Input[Dataset],
    model: Input[Dataset],
    metrics: Output[Metrics],
) -> None:
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    # データの読み込み
    X_test_df = pd.read_csv(X_test.path)
    y_test_df = pd.read_csv(y_test.path)
    model = joblib.load(model.path)

    # 予測の実行
    y_pred = model.predict(X_test_df)

    # メトリクスの計算
    metrics.log_metric("accuracy", accuracy_score(y_test_df, y_pred))
    metrics.log_metric("precision", precision_score(y_test_df, y_pred))
    metrics.log_metric("recall", recall_score(y_test_df, y_pred))


@pipeline(
    name="ml-pipeline",
    description="A simple ML pipeline with data preprocessing, model training and evaluation",
)
def ml_pipeline() -> None:
    # データの読み込み
    load_data_task = load_data()

    # データの前処理
    preprocess_task = preprocess_data(input_data=load_data_task.outputs["output_data"])

    # モデルのトレーニング
    train_task = train_model(
        X_train=preprocess_task.outputs["X_train"],
        y_train=preprocess_task.outputs["y_train"],
    )

    # モデルの評価
    evaluate_model(
        X_test=preprocess_task.outputs["X_test"],
        y_test=preprocess_task.outputs["y_test"],
        model=train_task.outputs["model"],
    )


if __name__ == "__main__":
    # パイプラインをコンパイル
    compiler.Compiler().compile(
        pipeline_func=ml_pipeline, package_path="ml_pipeline.yaml"
    )
