# Arima Mlfow Demo
The following repo is a demo for the mlFlow framework. We have used an ARIMA model to show the capabilities of the framework.
Please note there might be issues with the model but the goal is not to deliver a fully functional model.

## Getting started
First clone the repo and run it in your favourite IDE.
The entry point for the demo is from the unittest `arima_impl_tests.py`
Before starting the test make sure that your working directory points to the root of the project.
Also if you want to test the azure backend you will need to export the following environment variable:

* AZURE_STORAGE_ACCESS_KEY

You will need to use to key generated by the Azure Blob Storage account