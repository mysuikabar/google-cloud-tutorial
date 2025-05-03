Login to Google Cloud
```bash
gcloud auth login  # for gcloud
gcloud auth application-default login  # for terraform
```

Create infra
```bash
cd cloud_composer/infra
terraform init
terraform plan
terraform apply
```

Upload DAG to Cloud Composer
```bash
gcloud composer environments storage dags import \
--project $GOOGLE_CLOUD_PROJECT \
--location $GOOGLE_CLOUD_REGION \
--environment example-environment \
--source dag_sample.py
```

Check created DAG
```bash
gcloud composer environments run example-environment \
--project $GOOGLE_CLOUD_PROJECT \
--location $GOOGLE_CLOUD_REGION \
dags list
```
