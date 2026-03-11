# Sentiment Analysis from A14 road users comments

## Introduction

The user of the [A14 road](https://highwaysengland.co.uk/A14-Cambridge-to-Huntingdon-Improvement-Scheme-home) can report incident using an application. The goal of this challenge proposed by the organizers of the [Project:Hack5 hackathon](https://projectdataanalytics.uk/eventer/projecthack-5) and [Highways England](https://highwaysengland.co.uk/) is to perform sentiment analysis to obtain new insigts from the users comments and improve the user experiences on the application.

## Methods

I performed this analysis with Python. I used the TextBlob package to perform sentiment analysis on the A14 application users feedbacks. 

## Results

Using a Random Forest model, we can predict the nature of the user observation with an 84% accuracy. We could then use this model to prefill this variable on the form filled by the A14 users. This approach will reduce the amount of data to enter for the users and likely increase the form completion rate.