# Data_Science_portfolio_dev: A private repository to update and improve my data science portfolio

# Data Science portfolio
A collection of personal data sience and computational biology and projects.

## About me

I am an Immunologist with experience in molecular biology and data analysis with R and Python. As a researcher, I have developed strong skills in statistical analysis and communication. My research included analysis of complex genomic datasets of immune cell receptors and led to 10 articles in peer-reviewed journals. I have a passion for the use of machine learning in the drug discovery process.

## Contact Information
William Guesdon, London UK  
<a href="mailto:wguesdon@gmail.com">email</a>   
[Personal website](https://wguesdon.github.io//)  
[linkedIn profile](https://www.linkedin.com/in/william-guesdon/)  
[Kaggle progile](https://www.kaggle.com/wguesdon)

***
## Data Science projects

### [Predicting Heart Disease Risk with Random Forest]()
Heart disease is the leading cause of death in the United States ([1](https://www.cdc.gov/heartdisease/facts.htm)). To address this significant health issue, the Center for Disease Control (CDC) has a division dedicated to heart disease and stroke prevention ([2](https://www.cdc.gov/dhdsp/programs/spha/index.htm)). The CDC also recently started to use machine learning for prevention and diagnoses, which should be useful in identifying the population at risk ([3](https://www.electronicproducts.com/Programming/Software/The_CDC_uses_machine_learning_and_social_media_to_forecast_flu_outbreaks.aspx)).
Assisting diagnoses is an exciting and promising application of machine learning ([4](https://www.nature.com/articles/s41467-019-14225-8), [5](https://www.nature.com/articles/s41598-019-56889-8)), but the use of black box models is a potential issue ([6](https://www.thelancet.com/journals/lanres/article/PIIS2213-2600(18)304259/fulltext)). In this analysis, I used a decision tree open model to identify subjects with heart disease. The model achieved 80% accuracy, highlighting the potential of machine leering in assisting and accelerating the diagnosis of heart disease.

### [Determine factors influencing stroke]()
The goal of this assignment is to use statistical learning to identify the combination of the features that are more likely to be associated with stroke. For this analysis, I first performed an exploratory data analysis and feature engineering. As often seen in health-related datasets, the proportion of patients with stroke is low, resulting in a severely unbalanced dataset. To compensate for the unbalance, I used the ROSE package to artificially over sample the number of strokes observations.  
I then used a dimension reduction technique adapted to mixed datasets of continuous and categorical variables with the FAMD package. This visualization allowed me to identify the features most likely associated with the risk of developing a stroke.
Using these selected variables, I built a logistic regression model to evaluate the contribution of each feature to the risk of developing a stroke. The logistic regression model identified the Age, Average Glucose Level, Smoking, Hypertension, and Heart Diseases features as the most likely risk factors of developing a stroke. Age was the principal risk factor in this study.

### [Sentiment analysis and incident prediction (Project Hack 5)]()
The users of the A14 road can report incidents using an application. The goal of this challenge proposed by the organizers of the Project: Hack5 hackathon is to perform sentiment analysis to obtain new insights from the user's comments and improve the user experiences on the application.
The sentiment analysis was performed using the TextBlog library in python. Based on the sentiment analysis, an incident prediction tool was built using a Random Forest algorithm. The model can predict the type of incident with an accuracy of 83% and could be used to add a questionnaire auto-completion functionality in the application. 

### [Food Safety Agency Hackathon: young population survey analysis]()  
The Food Standards Agency (FSA) is an independent government department working across England, Wales and Northern Ireland to protect public health and consumers’ wider interest in food. The FSA is responsible for making sure food is safe and what it says it is.  
On Sat, November 23, 2019 FSA and Pivigo organized a [Hackathon](https://www.eventbrite.com/e/food-standards-agency-data-science-hackathon-tickets-77135950705?utm_source=eventbrite&utm_medium=email&utm_campaign=reminder_attendees_48hour_email&utm_term=eventname&ref=eemaileventremind#) to analyse food survey questionaires.  

### [NYC Airbnb: house price prediction]()
For this project, I explored a dataset of Airbnb properties in NYC. I used Python and the Seaborn library to visualize the data. The examination of the dataset highlights the large variability of the price distribution and the expected impact of the neighborhood and room type on price. To model the property price based on the independent variables, I used the scikit-learn library to implement multiple linear regression and random forest regressors. Given the limited number of independent variables on the dataset, the model accuracy was limited, especially for the higher-priced properties. A more elaborated model could be built using neural network and natural language processing analysis of the properties reviews to improve the model efficiency.

### [Melbourne Housing Market: Multiple Linear Regression]()
For this project, as for the NYC Airbnb analysis, I used Python and the scikit-learn library to predict the price of houses based on the independent variables. Because there were more variables in this dataset and because several factors were linearly correlated to the price, a multiple linear regression model performed well.

***
## Computational Biology projects
### [B cells repertoire analysis]()
Systemic lupus erythematosus (SLE) is an autoimmune disease in which the B lymphocytes produce pathogenic auto-antibodies targeting healthy tissues. The exact causes of the diseases are unknown but involves a combination of genetic and environmental factors. In this project I used the AntibodyMap database to extract the heavy chains of healthy subject and SLE Patients. My interest was particularly to compare the IgM heavy chains repertoire between healthy control and SLE patients. I used the Immcantation pipeline developed by Steven Kleinstein’s team and my owns custom scripts to compare the B cell repertoire of SLE patient to healthy controls.

### [10x Genomics: identify cells subsets]()
10X Genomics: identify cells subsets barcodes

### [Ubuntu Server]()
The steps used to create the Ubuntu 18.04 server used for B cells Receptor clonal analysis.


