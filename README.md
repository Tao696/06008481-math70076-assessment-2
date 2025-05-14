# 06008481-math70076-assessment-2
Analysis of Survival Data for Patients Undergoing Allogeneic Hematopoietic Cell Transplantation (HCT)

# HCT Patient Survival Analysis Project Documentation

## Project Overview

This project aims to analyze survival data of patients undergoing allogeneic hematopoietic cell transplantation (HCT), construct predictive models to assess patient survival probabilities, and identify key factors affecting survival. The project is based on a survival analysis dataset consisting of 28,800 patients and 60 features, with right-censored data (46% of patients are still alive at the end of the study). By applying Kaplan-Meier and Survival-Cox models, and integrating XGBoost, CatBoost, and LightGBM fitting models, five composite models were developed, their predictive performance compared, and feature importance analyzed. The final results can serve as a reference for clinical decision-making.

## Project Structure

The project structure is as follows:

- **02-data**  
  - Contains the original dataset, data sources, and intermediate datasets for data preprocessing and analysis.  

- **03-src**  
  - Contains the core code for data preprocessing, model construction, and feature analysis.  

- **04-tests**  
  - Contains test scripts for verifying code functionality and model performance.  

- **05-analyses**  
  - Contains the analysis scripts used during the analysis process.  

- **06-outputs**  
  - Contains the final model output results, such as feature importance rankings and C-index scores.  

- **07-reports**  
  - Contains project reports and summary documents, including analysis results and reflections.  

