# Ph.D. Stipends by research topic and universities

## Introduction

This project aims to clean up and analyze the data set of Ph. D. students salaries by universities and departments over time.
I performed the analysis with  R and the tidyverse libraries.
For the data cleaning, I excluded the variables containing a majority of missing values, combined similar departments, and separated the universities per location.
The significant highlights from the analysis are:  

* The majority of responders are from the USA, data collection from non-USA universities started in 2013
* The students stipend and living wages are higher in the USA.
* The stipends do not increase with experience
* The Ph.D. stipends had a significant decrease around the 2008 crisis.
* The Ph.D. stipends are not equal between departments.

## Methods

The Data set, version 7, was downloaded on [Kaggle](https://www.kaggle.com/paultimothymooney/phd-stipends/data) on the 2020-07-12.
The analysis was performed with R. The data cleaning was performed with Dplyr and the visualization with Ggplot2.

## Results

* The Overall Pay and Living Wage ratio is higher in the USA.
* PhD students Living Wage ratios have decreased internationally since 2013.
* Living Wages ratios started to increase again after 2015.
* Business students are the best paid in the USA.
* English students have the lowest paid in the USA.

## Future studies

The following areas could be explored further:

 * compare LW ratio to professional salaries for each given department.
 * compare LW ratio to average student debt by countries.
 * compare salaries to inflation in the USA.