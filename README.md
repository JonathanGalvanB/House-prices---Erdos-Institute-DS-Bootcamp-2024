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

## Approach:



### Model #1:


### Model #2:




### Results


## Future Improvements


## Conclusion
