# Capstone Project - Azure Machine Learning Engineer

*TODO:* Write a short introduction to your project.

As it is our final project in Udacity program, we chose to apply what we learned to a forecasting model. For this purpose we took our inspiration from this [git repository](https://github.com/microsoft/forecasting/blob/86b421b71826b92e47c3e3cb6cdcbf7ff4a63b90/examples/grocery_sales/README.md)

The project is about the Orange Juice dataset. We use two methods to select the best forecasting model. The first method consists of tuning the scikit learn model LightGBM. The second method is to configure AutoML with a task of forecasting. We refer to the primary metric Mean Absolute Percentage Error (MAPE). The model that obtains the lowest of MAPE is deployed using Python SDK.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

We use the Orange Juice dataset taken from the R package [Bayesian Inference for Marketing/Micro-Econometrics](https://cran.r-project.org/web/packages/bayesm/index.html) (bayesm). It represents weekly sales of refrigerated orange juice at 83 stores. Also, it contains demographic information on those stores. To be able to use this dataset, we execute an R script to convert the orangeJuice.rda located in "./starter_file/ojdata" into two files :

  - "xy.csv" : Weekly sales of refrigerated orange juice at 83 stores. It has 106139 rows and 19 columns.
  
              $store : store number
              
              $brand : brand indicator
              
              $week : week number
              
              $logmove : log of the number of units sold
              
              $constant : a vector of 1s
              
              $price# : price of each brand. It is 11 columns.
              
              $deal : in-store coupon activity
              
              $feature : feature advertisement
              
              $profit : profit obtained
              
  - "storedemo.csv" : Demographic information on the 83 stores. It has 83 rows and 13 columns. 
  
              $STORE : store number
              
              $AGE60 : percentage of the population that is aged 60 or older
              
              $EDUC : percentage of the population that has a college degree
              
              $ETHNIC : percent of the population that has different ethnicity
              
              $INCOME : median income
              
              $HHLARGE : percentage of households with 5 or more persons
              
              $WORKWOM : percentage of women with full-time jobs
              
              $HVAL150 : percentage of households worth more than $150,000
              
              $SSTRDIST : distance to the nearest warehouse store
              
              $SSTRVOL : ratio of sales of this store to the nearest warehouse store
              
              $CPDIST5 : average distance in miles to the nearest 5 supermarkets
              
              $CPWVOL5 : ratio of sales of this store to the average of the nearest five stores

We use the two files to generate two directories "train" and "test". The "train" directory includes "train.csv" and "auxi.csv" with "train.csv" containing the historical sales up to week 135 (the time we make forecasts) and "auxi.csv" containing price/promotion information up until week 138. We assume that future price and promotion information up to a certain number of weeks ahead is predetermined and known. On the other hand, the directory "test" has the sales of each product in week 137 and 138. 

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

In this dataset, we start from week 40 to week 138. The "train.csv" file contains historical sales up to week 135. Our task is to forecast the sales of the 157 and 158 week. The week 136 is represented as a gap to leave time for planning inventory as in real life.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

We have the dataset as csv files in our directory "ojdata". in Azure Studio, we register the files of "train" directory under their respective names "train" and "auxi". This registered datasets is used for the hypertuning. For the test file, we use it after we deploy our model. 

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

We wanted to convert our AutoML model to ONNX format. However, due to an incompatibility reason for a forecasting task, we were unable to set "True" the parameter "enable_onnx_compatible_models". Here is a proof of the error. 
<img src="./starter_file/screenshots/config_onnx.PNG">
<img src="./starter_file/screenshots/onnx_error.PNG">
