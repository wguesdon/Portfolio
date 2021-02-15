# Data Science portfolio
A collection of personal data science and computational biology projects.

## About me

I am a Senior Scientist with experience in immunology, molecular biology and data analysis with R and Python. As a researcher, I have developed strong skills in statistical analysis and communication. My research included analysing complex genomic datasets of immune cell receptors and led to 13 articles in peer-reviewed journals. I have a passion for the use of machine learning in the drug discovery process.

## Contact Information
William Guesdon, Manchester UK  
<a href="mailto:wguesdon@gmail.com">email</a>   
[Personal website](https://wguesdon.github.io//)  
[linkedIn](https://www.linkedin.com/in/william-guesdon/)  
[Kaggle](https://www.kaggle.com/wguesdon)

***
## Data Science projects

### [Discord Server Messages Analysis](https://github.com/wguesdon/Data_Science_portfolio/tree/master/Data_Science_projects/Discord_Server_Messages_Analysis)

[66DaysOfData](https://www.66daysofdata.com/) is a challenge to learn data science by committing to work at least 5 min every day and share your progress. 
The goal of this project is to analyse the messages of the 66DaysOfData discord server. The first analysis focuses on the Introduction and Progress channels. 

### [Sentiment analysis and incident prediction (Project Hack 5)](https://github.com/wguesdon/Data_Science_portfolio/tree/master/Data_Science_projects/Sentiment_Analysis_from_A14_Observation_Report)
The users of the A14 road can report incidents using an application. The goal of this challenge proposed by the organizers of the Project: Hack5 hackathon is to perform sentiment analysis to obtain new insights from the user's comments and improve the user experiences on the application.
The sentiment analysis was performed using the TextBlog library in python. Based on the sentiment analysis, an incident prediction tool was built using a Random Forest algorithm. The model can predict the type of incident with an accuracy of 83% and could be used to add a questionnaire auto-completion functionality in the application. 

### [Predicting Heart Disease Risk with Random Forest](https://github.com/wguesdon/Data_Science_portfolio/tree/master/Data_Science_projects/Heart_disease_risk_analysis)
Heart disease is the leading cause of death in the United States ([1](https://www.cdc.gov/heartdisease/facts.htm)). To address this significant health issue, the Center for Disease Control (CDC) has a division dedicated to heart disease and stroke prevention ([2](https://www.cdc.gov/dhdsp/programs/spha/index.htm)). The CDC also recently started to use machine learning for prevention and diagnoses, which should be useful in identifying the population at risk ([3](https://www.electronicproducts.com/Programming/Software/The_CDC_uses_machine_learning_and_social_media_to_forecast_flu_outbreaks.aspx)).
Assisting diagnoses is an exciting and promising application of machine learning ([4](https://www.nature.com/articles/s41467-019-14225-8), [5](https://www.nature.com/articles/s41598-019-56889-8)), but the use of black box models is a potential issue ([6](https://www.thelancet.com/journals/lanres/article/PIIS2213-2600(18)304259/fulltext)). In this analysis, I used a decision tree open model to identify subjects with heart disease. The model achieved 80% accuracy, highlighting the potential of machine leering in assisting and accelerating the diagnosis of heart disease.

### [Determine Risks Factors Associated with Strokes](https://github.com/wguesdon/Data_Science_portfolio/tree/master/Data_Science_projects/Determine_Risks_Factors_Associated_with_Strokes)
The goal of this assignment is to use statistical learning to identify the combination of the features that are more likely to be associated with stroke. For this analysis, I first performed an exploratory data analysis and feature engineering. As often seen in health-related datasets, the proportion of patients with stroke is low, resulting in a severely unbalanced dataset. To compensate for the unbalance, I used the ROSE package to artificially over sample the number of strokes observations.  
I then used a dimension reduction technique adapted to mixed datasets of continuous and categorical variables with the FAMD package. This visualization allowed me to identify the features most likely associated with the risk of developing a stroke.
Using these selected variables, I built a logistic regression model to evaluate the contribution of each feature to the risk of developing a stroke. The logistic regression model identified the Age, Average Glucose Level, Smoking, Hypertension, and Heart Diseases features as the most likely risk factors of developing a stroke. Age was the principal risk factor in this study.

### [NYC Airbnb: house price prediction](https://github.com/wguesdon/Data_Science_portfolio/tree/master/Data_Science_projects/NYC_Airbnb)
For this project, I explored a dataset of Airbnb properties in NYC. I used Python and the Seaborn library to visualize the data. The examination of the dataset highlights the large variability of the price distribution and the expected impact of the neighborhood and room type on price. To model the property price based on the independent variables, I used the scikit-learn library to implement multiple linear regression and random forest regressors. Given the limited number of independent variables on the dataset, the model accuracy was limited, especially for the higher-priced properties. A more elaborated model could be built using neural network and natural language processing analysis of the properties reviews to improve the model efficiency.

### [Melbourne Housing Market: Multiple Linear Regression](https://github.com/wguesdon/Data_Science_portfolio/tree/master/Data_Science_projects/Melbourne_Housing_Market)
For this project, as for the NYC Airbnb analysis, I used Python and the scikit-learn library to predict the price of houses based on the independent variables. Because there were more variables in this dataset and because several factors were linearly correlated to the price, a multiple linear regression model performed well.

### [Ph.D. Stipends by research topic and universities](https://github.com/wguesdon/Data_Science_portfolio/tree/master/Data_Science_projects/PhD_Stipends_by%20research_topic_and_universities)

This project was proposed by the [Data Scientist Syndicate](https://cheekyscientist.com/career-programs/data-scientist-syndicate/) facebook group.

This project aims to clean up and analyze the data set of Ph. D. students salaries by universities and departments over time.
I performed the analysis with  R and the tidyverse libraries.
For the data cleaning, I excluded the variables containing a majority of missing values, combined similar departments, and separated the universities per location.
The significant highlights from the analysis are:  

* The majority of responders are from the USA, data collection from non-USA universities started in 2013
* The students stipend and living wages are higher in the USA.
* The stipends do not increase with experience
* The Ph.D. stipends had a significant decrease around the 2008 crisis.
* The Ph.D. stipends are not equal between departments.

### [Analyze BrainPoste Blog traffic](https://github.com/wguesdon/Data_Science_portfolio/tree/master/Data_Science_projects/Blog_Google_Analytics)

This project was proposed by the [Data Scientist Syndicate](https://cheekyscientist.com/career-programs/data-scientist-syndicate/) facebook group.

Kasey Hemington runs BrainPost with a fellow PhD friend, Leigh Christopher as a way to keep in touch with her scientific roots while working as a data scientist!

The goal of this project is to answer the following question:

* What content (or types of content) is most popular (what are patterns we see in popular content) and is different content popular amongst different subgroups (e.g. by source/medium)?

* Where are people visiting from (source-wise)?

### [Food Safety Agency Hackathon: young population survey analysis](https://github.com/wguesdon/Data_Science_portfolio/tree/master/Data_Science_projects/FSA_Hackathon)  
The Food Standards Agency (FSA) is an independent government department working across England, Wales and Northern Ireland to protect public health and consumers’ wider interest in food. The FSA is responsible for making sure food is safe and what it says it is.  
On Sat, November 23, 2019 FSA and Pivigo organized a [Hackathon](https://www.eventbrite.com/e/food-standards-agency-data-science-hackathon-tickets-77135950705?utm_source=eventbrite&utm_medium=email&utm_campaign=reminder_attendees_48hour_email&utm_term=eventname&ref=eemaileventremind#) to analyse food survey questionaires.  

### [Visualizing Inequalities in Life Expectancy](https://github.com/wguesdon/Data_Science_portfolio/tree/master/Data_Science_projects/Visualizing_Inequalities_in_Life_Expectancy)

This simple project proposed by DataCamp aims to visualize the inequalities in life expectancy among countries using R.
I performed the data cleaning and wrangling with the Dplyr package and the visualization with the Ggplot2 package.

***
## Computational Biology projects
### [B cells repertoire analysis](https://github.com/wguesdon/Data_Science_portfolio/tree/master/Computational_Biology_projects/B_cell_repertoire_analysis)
Systemic lupus erythematosus (SLE) is an autoimmune disease in which the B lymphocytes produce pathogenic auto-antibodies targeting healthy tissues. The exact causes of the diseases are unknown but involves a combination of genetic and environmental factors. In this project I used the AntibodyMap database to extract the heavy chains of healthy subject and SLE Patients. My interest was particularly to compare the IgM heavy chains repertoire between healthy control and SLE patients. I used the Immcantation pipeline developed by Steven Kleinstein’s team and my owns custom scripts to compare the B cell repertoire of SLE patient to healthy controls.

### [Ubuntu Server](https://github.com/wguesdon/Data_Science_portfolio/tree/master/Computational_Biology_projects/Ubuntu_Server)
The steps used to create the Ubuntu 18.04 server used for B cells Receptor clonal analysis.