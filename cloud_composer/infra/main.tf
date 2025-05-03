// reference: https://cloud.google.com/composer/docs/composer-3/terraform-create-environments?hl=ja

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

resource "google_project_service" "composer_api" {
  provider = google-beta
  project  = var.project_id
  service  = "composer.googleapis.com"
  // Disabling Cloud Composer API might irreversibly break all other
  // environments in your project.
  disable_on_destroy = false
  // this flag is introduced in 5.39.0 version of Terraform. If set to true it will
  //prevent you from disabling composer_api through Terraform if any environment was
  //there in the last 30 days
  check_if_service_has_usage_on_destroy = true
}

resource "google_service_account" "custom_service_account" {
  provider     = google-beta
  account_id   = "custom-service-account"
  display_name = "Example Custom Service Account"
}

resource "google_project_iam_member" "custom_service_account" {
  provider = google-beta
  project  = var.project_id
  member   = format("serviceAccount:%s", google_service_account.custom_service_account.email)
  // Role for Public IP environments
  role = "roles/composer.worker"
}

resource "google_composer_environment" "example_environment" {
  provider = google-beta
  name     = "example-environment"
  config {
    software_config {
      image_version = "composer-3-airflow-2.10.2-build.7"
    }
    node_config {
      service_account = google_service_account.custom_service_account.email
    }
  }
}