# House prices - Erdös Institute DS Bootcamp 2024
 House Prices project repository 
 Group Members: Krescens Kok and Jonathan Galvan

 ## Project Overview:

 The housing market consistently changes, especially from year to year and there are many other features of a house that can impact the cost of a home. 
 We decided to see whether we could predict how much a house will sell based on features such as the quality, number of bedrooms, square footage, fireplaces, etc...

 This information is useful for:
 - Realtors
 - Home buyers
 - Home sellers
 - Investors

## Project Goal:
Determine whether we can predict how much a house will sell based on numerous features.

## Dataset:
Our dataset consists of houses in Ames, Iowa that were sold throughout numerous years. The dataset includes 79 characteristics/columns for 1,460 houses, including both numerical and categorical variables.
The characteristics could be grouped into separate categories:
- Geographical: Zoning classification, Neighborhood, Lot shape, Land slope, etc...
- Size Measures: Lot frontage footage, liveable square footage, square footage of the basement, etc...
- Quality Measures: Overall condition, roof material, exterior covering, utilities present, etc...
- Special Features: Fireplaces, pools, fences, elevator, tennis courts, etc...
- Status of Garage and Basement: Number of cars fitting in the garage, number of bathrooms in the basement, square footage of the finished basement, etc...


## Data Pre-Processing:
Before starting with a modeling approach, it was necessary to clean up our data by removing null values by imputing them with other values, converting categorical variables into dummy variables, and removing outliers by scaling the data.

#### **Imputing:**
There were a lot of categorical variables where a missing value meant 'None' or the feature was not present in the house. So, we decided to fill those null values in with 'None'.

![image](https://github.com/JonathanGalvanB/House-prices---Erdos-Institute-DS-Bootcamp-2024/assets/71037216/8b7f33bd-c34e-462b-b88c-43b82b6b1106)

For the numerical variables, we decided to use the KNNImputer from the sklearn library to impute the closest values that would make sense.

![image](https://github.com/JonathanGalvanB/House-prices---Erdos-Institute-DS-Bootcamp-2024/assets/71037216/6c1f6e52-0ddd-4c20-b6de-860908a64953)

#### Dummy Variables:
For converting our categorical variables to dummy variables, we decided to use one-hot encoding and label encoding. 
We used 2 different encodings for this process:

The first one was pd.get_dummies():

![image](https://github.com/JonathanGalvanB/House-prices---Erdos-Institute-DS-Bootcamp-2024/assets/71037216/bc381ba7-b5f4-4282-9000-bc067f03cefe)

The alternative was using the OrdinalEncoder function from the sklearn library:

![image](https://github.com/JonathanGalvanB/House-prices---Erdos-Institute-DS-Bootcamp-2024/assets/71037216/9b1a69ea-d89a-4068-8d92-ff5b6c5b6878)

#### Outliers:

To handle outliers, we decided to normalize our data by using the MinMaxScaler function from the sklearn library:

![image](https://github.com/JonathanGalvanB/House-prices---Erdos-Institute-DS-Bootcamp-2024/assets/71037216/eb2f3892-c4de-450c-9c86-2f99b33b49a0)

More details about our pre-processing can be found here: [Data-Cleaning](https://github.com/JonathanGalvanB/House-prices---Erdos-Institute-DS-Bootcamp-2024/blob/main/Notebooks/train_data_cleaning.ipynb)

## Data Visualization:

When visualizing data, we thought it would be important to see which features had a collinearity with each other, which would lead us to removing one of those features. We decided to use a correlation heatmap to help us better visualize it.

![image](https://github.com/JonathanGalvanB/House-prices---Erdos-Institute-DS-Bootcamp-2024/assets/71037216/85be54bf-b28a-4ce3-91de-779113cf3272)

We then were able to calculate the correlation values by using the following function:

![image](https://github.com/JonathanGalvanB/House-prices---Erdos-Institute-DS-Bootcamp-2024/assets/71037216/f80b0ac4-695d-43d9-b54f-157f4873e967)

By using these values, we were able to decide which features we could either drop or combine in order to reduce our dimensionality.

![image](https://github.com/JonathanGalvanB/House-prices---Erdos-Institute-DS-Bootcamp-2024/assets/71037216/5b41f03c-8dda-434e-bdb7-f0bd146937a7)

After cleaning up the data a little more, we created a few time series plots to visualize how the years or months may have had an impact on the `SalePrice`

Looking at the Sale Price vs Month Sold, there is no concrete pattern of the month affecting the `SalePrice`. However, we do see that in the Spring/Summer months, we see a few spikes in the `SalePrice`. This makes sense as this is known to be the active season of buyers looking and buying houses, where as the winter months are considered the dead season for the housing market.

![image](https://github.com/JonathanGalvanB/House-prices---Erdos-Institute-DS-Bootcamp-2024/assets/71037216/54460901-e5fe-4134-8427-2c6c047ec382)

Looking at the Sale Price vs Year Sold, there is also no pattern of the years affecting the `SalePrice`. However, again, we do see that there is a max around 2008. 

![image](https://github.com/JonathanGalvanB/House-prices---Erdos-Institute-DS-Bootcamp-2024/assets/71037216/cbcecc1e-0870-4b80-9863-b7a33bfcf8d6)

Lastly, we wanted to visualize whether certain features would have an affect on the `SalePrice`.

We ploted the Sale Price vs Total Condition, and it seems that as the `Total_Condition` score increases, the `SalePrice` also increases, resulting in a positive correlation.

![image](https://github.com/JonathanGalvanB/House-prices---Erdos-Institute-DS-Bootcamp-2024/assets/71037216/bee592e3-c456-4aee-b999-c4da1ffe1bf2)

We decided to try the same thing with the Overall Rating of a house, however, we did not see the same pattern.

![image](https://github.com/JonathanGalvanB/House-prices---Erdos-Institute-DS-Bootcamp-2024/assets/71037216/4084bc43-8cfa-4170-8ccc-b5f0ad2c9a13)

We then thought that the total square footage of a house may impact the `SalePrice`, but looking at the graph, there is no correlation between the 2 features.

![image](https://github.com/JonathanGalvanB/House-prices---Erdos-Institute-DS-Bootcamp-2024/assets/71037216/a374e308-480d-490a-a9d0-e6e63e601f1d)

More details about our pre-processing can be found here: [Data-Visualization](https://github.com/JonathanGalvanB/House-prices---Erdos-Institute-DS-Bootcamp-2024/blob/main/Notebooks/data_visualization.ipynb)


## Approach:

Originally, we thought using an ARIMA model (time series) would be benificial for predicting the `SalePrice`, however, once trying it, we realized the data did not have evenly spaced date intervals so we were not able to treat it as a time series. 

Furthermore, looking at the graph of the Sale Price vs Year Sold, we can confirm that there is no pattern.

![image](https://github.com/JonathanGalvanB/House-prices---Erdos-Institute-DS-Bootcamp-2024/assets/71037216/1397c517-697e-4440-91d9-586c94ee3691)

### Model #1: Random Forest Regressor

Our next thought was to predict the logarithm of the house prices to uniformize the contribution of individual errors. 

We found that using a RandomForestRegressor resulted in a mean squared error of .00357 and the $r^2$ value was .868

![image](https://github.com/JonathanGalvanB/House-prices---Erdos-Institute-DS-Bootcamp-2024/assets/71037216/f91c6567-40d8-4947-9c02-0e6f4c3959c0)

### Model #2: XGBoost Regressor

We also tried using the XGBoost model, and also resulted in pretty good metrics with a mean squared error of .00335 and an $r^2$ value of .876

![image](https://github.com/JonathanGalvanB/House-prices---Erdos-Institute-DS-Bootcamp-2024/assets/71037216/223c7c8d-906b-4d2e-8880-959213256783)

### Model #3: Linear Regression

Lastly, we used a linear regression to use for comparison and found that the mean squared error was .0089 and an $r^2$ value of .844

![image](https://github.com/JonathanGalvanB/House-prices---Erdos-Institute-DS-Bootcamp-2024/assets/71037216/daa17bd7-92b3-4fa4-ac26-2a89531c8bb3)

### Results

Looking at the metrics of all 3 models, we concluded that the XGBoost Regressor was the best, resulting in the lowest MSE and highest $r^2$ value. 

We also were able to determine the most important features that would help with this model, by looking at the weight. The top 3 important features include:
- GrLivArea
- Overall_Rating
- LotFrontage

Logically, these variables make sense that it would be important in predicting a house price since usually, the bigger the house/lot area, the higher the sale price.

![image](https://github.com/JonathanGalvanB/House-prices---Erdos-Institute-DS-Bootcamp-2024/assets/71037216/76a68213-b486-4047-8181-1b1f9e5a22bf)


## Future Improvements

A future improvement that would be beneficial is creating more visualizations to figure out which features may be important. Also, looking more at the graphs, we should have known that the ARIMA model would not be the best model to use since we did not see any sort of pattern. 

## Conclusion

Back to our research question and project goal of can we determine whether we can predict how much a house will sell based on numerous features? We determined that we can indeed predict the `SalePrice` based on about 30 features. Using all the features with the Linear Regression resulted in a larger mse, indicating that there was overfitting involved. 


