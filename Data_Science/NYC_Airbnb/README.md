# NYC Airbnb: EDA, Visualization, Regression

## Introduction

This project aims to explore the Airbnb properties in New York City.

## Methods

I performed the analysis with Python. I used a RandomForestRegressor included in scikit-learn model to predict the price of the properties. 

## Results

* The properties have large differences in prices. 
* Separating the dataset by price categories is useful for the analysis.
* The most interesting variables regarding price prediction are:
  * Location
  * Room type
  * calculated_host_listings_count
  * Number of reviews
* Price prediction models are not performing well. The best score is 0.55
* Predictions are more accurate for a price under $175 (75% of the dataset)
* Using categorical encoded data did not improve the model.
