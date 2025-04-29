import os

from dotenv import load_dotenv
from google.cloud import aiplatform

# .envファイルから環境変数を読み込む
load_dotenv()

# 環境変数からプロジェクトIDとリージョンを取得
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
REGION = os.getenv("GOOGLE_CLOUD_REGION")
PIPELINE_ROOT = f"gs://{PROJECT_ID}-vertex-pipelines"


def run_pipeline() -> None:
    # Vertex AIの初期化
    aiplatform.init(project=PROJECT_ID, location=REGION)

    # パイプラインジョブの作成と実行
    job = aiplatform.PipelineJob(
        display_name="ml-pipeline-tutorial",
        template_path="ml_pipeline.yaml",
        pipeline_root=PIPELINE_ROOT,
    )

    # パイプラインの実行
    job.submit()


if __name__ == "__main__":
    if not PROJECT_ID:
        print("エラー: GOOGLE_CLOUD_PROJECT環境変数が設定されていません")
        print("環境変数は.envファイルで設定してください")
        exit(1)

    print(f"プロジェクトID: {PROJECT_ID}")
    print(f"リージョン: {REGION}")
    print(f"パイプラインルート: {PIPELINE_ROOT}")

    run_pipeline()
    print(
        "パイプラインの実行を開始しました。進行状況はVertex AI Pipelinesのコンソールで確認できます。"
    )
