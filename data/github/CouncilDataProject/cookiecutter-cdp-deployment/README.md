# https://github.com/CouncilDataProject/cookiecutter-cdp-deployment

```console
{{ cookiecutter.hosting_github_repo_name }}/SETUP/README.md:1. Once the ["Infrastructure" GitHub Action Successfully Completes]({{ cookiecutter.hosting_github_url }}/actions?query=workflow%3A%22Infrastructure%22) request a quota increase for `compute.googleapis.com/gpus_all_regions`.
{{ cookiecutter.hosting_github_repo_name }}/SETUP/README.md:    [Direct Link to Quota](https://console.cloud.google.com/iam-admin/quotas?project={{ cookiecutter.infrastructure_slug }}&pageState=(%22allQuotasTable%22:(%22f%22:%22%255B%257B_22k_22_3A_22Metric_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22compute.googleapis.com%252Fgpus_all_regions_5C_22_22_2C_22s_22_3Atrue_2C_22i_22_3A_22metricName_22%257D%255D%22)))
{{ cookiecutter.hosting_github_repo_name }}/SETUP/README.md:    -   Click the checkbox for the "GPUs (all regions)"
{{ cookiecutter.hosting_github_repo_name }}/SETUP/README.md:        -   You can request more or less than `2` GPUs, however we have noticed that a

```
