# Cloud Run Jobs Tutorial

Login to Google Cloud
```bash
gcloud auth login
```

Build the Docker image

```bash
gcloud builds submit --tag gcr.io/$GOOGLE_CLOUD_PROJECT/cloud-run-jobs-sample
```

Deploy the Cloud Run job
```bash
gcloud run jobs create cloud-run-jobs-sample \
    --image gcr.io/$GOOGLE_CLOUD_PROJECT/cloud-run-jobs-sample \
    --region $GOOGLE_CLOUD_REGION
```

Run the Cloud Run job
```bash
gcloud run jobs execute cloud-run-jobs-sample --region $GOOGLE_CLOUD_REGION
```
