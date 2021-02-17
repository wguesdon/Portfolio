# Determine Risks Factors Associated with Strokes

## Introduction

This project aims to identify the factors most likely to be associated with the risk of developing a stroke. 

## Methods

I performed the analysis with R. The dataset contains a majority of healthy patients, and therefore I used the ROSE package to balance the dataset. I completed the MCA dimension reduction using the factoextra package.

## Results

Using dimension reduction and supervised machine learning, I was able to identify 5 variables as the principal risk factors of developing a stroke:

* Age
* Average Glucose Level
* Ever Smoked
* Hypertension
* Heart Disease

These features could be used to build a machine learning model of stroke risk using logistic regression or random forest algorithms. Although given the limited number of initial observations, such a model is more at risk to overfit even after correction for sample unbalance.
