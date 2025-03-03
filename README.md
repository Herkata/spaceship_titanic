# Spaceship Titanic Kaggle Competition

This repository contains code and resources to showcase my submission to the above mentioned competition.

## Table of Contents

- [Overview](#overview)
  - [Dataset](#dataset)
  - [Repository Structure](#repository-structure)
- [EDA](#eda)
  - [Statistical overview of the data](#statistical-overview-of-the-data)
  - [Plots](#plots)
  - [Validating duplicated names](#validating-duplicated-names)
- [Data Preprocessing](#data-preprocessing)
  - [Handling Missing Values](#handling-missing-values)
    - [HomePlanet](#homeplanet)
    - [CryoSleep](#cryosleep)
- [Statistical Inference](#statistical-inference)
- [Model Building](#model-building)
  - [Model Selection](#model-selection)
- [Results](#results)
- [Leaderboard](#leaderboard)
- [Credit](#credit)
- [License](#license)

## Overview

The goal of this competition is to predict which passengers were transported to an alternate dimension during the collision of the Spaceship Titanic.

Access the competition page [here](https://www.kaggle.com/competitions/spaceship-titanic/overview).

### Dataset

**Features:**
- **PassengerId** - str: A unique Id for each passenger. Each Id takes the form `gggg_pp` where `gggg` indicates a group the passenger is travelling with and `pp` is their number within the group. People in a group are often family members, but not always.
- **HomePlanet** - str: The planet the passenger departed from, typically their planet of permanent residence.
- **CryoSleep** - bool: Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
- **Cabin** - str: The cabin number where the passenger is staying. Takes the form `deck/num/side`, where side can be either `P` for Port or `S` for Starboard.
- **Destination** - str: The planet the passenger will be debarking to.
- **Age** - float: The age of the passenger.
- **VIP** - bool: Whether the passenger has paid for special VIP service during the voyage.
- **RoomService, FoodCourt, ShoppingMall, Spa, VRDeck** - float: Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
- **Name** - str: The first and last names of the passenger.

**Target:**

- **Transported** - bool: Whether the passenger was transported to another dimension.

As the target is categorical, we are facing a binary classification problem.

### Repository Structure

- `spaceship_titanic_data/`: Contains the dataset files.
- `space_titanic.ipynb`: Jupyter notebook for data exploration and model building.
- `helper.py`: Additional code moved from the notebook for better readability.
- `requirements.txt`: List of required Python libraries.

```sh
pip install -r requirements.txt
```

## EDA

The notebook `spaceship_titanic.ipynb` contains exploratory data analysis and visualization of the dataset from different perspectives.

### Statistical overview of the data

- Data types
- Counts
- General description with numerical statistics (min-max, mean, unique values etc.)
- Missing values
- Duplicates
  
### Plots

- Numerical distributions
- Proportions in categorical variables
- Correlation matrix
- Pairplot to discover hidden patterns connecting features
- Split comparison within features to discover effect on target
- Grouping by multiple features to find connections for imputation
 
### Validating duplicated names

There are no direct duplicates in the data, but still multiple passengers by the same name, I decided to check if there is some sort of discrepancy in this field, as we are talking about alternate dimensions and such, we might as well have the same person twice on board. :)

The passengers with the same name get compared based on numerical distance in their features and jaccard similarity index for categorical features.

Result: final similarity score ranging from 0 to 1.

In this case unneccessary, useful for bigger datasets where manual handling is not an option and removing records does not result in significant information loss.

## Data Preprocessing

The most important steps in data preprocessing include:

- **Handling missing values**
- Encoding categorical variables
- Feature engineering
- Feature scaling

### Handling Missing Values

The dataset contains missing values in all columns except for `Passenger_id` and the target `Transported` itself.

The biggest challenge of this project is the correct imputation in all of the columns based on hidden patterns waiting to be discovered during EDA.

Based on [SMORES](<https://forums.ni.com/t5/Random-Ramblings-on-LabVIEW/SMORES-SMURF-or-SCRFFMRDM/ba-p/3488988>) (Scalable, Modular, Optimized, Reusable, Extensible, Simplified) principles, I decided to use a pipeline for these steps, as it will most possibly improve performance when used on unlabeled test data with missing values.

#### `HomePlanet`

Imputing missing `HomePlanet` values based on the `Cabin`, `Name` and the `Destination` features, as they both show a clear tendency to have distinct groups depending on HomePlanet.

- Cabin: there are certain decks "reserved" for certain planets
- Name: passengers with the same last name come from the same planet
- Destination: the rest of the missing values filled with the most common HomePlanet for the Destination of the traveller.

#### `CryoSleep`

Imputing based on the total spending:

- Spent money on board -> not asleep -> False
- True otherwise

#### `Cabin`

If the passenger was travelling with a group, use their cabin number.

If the passenger travels solo:
- I appointed them to the exlusive Deck for their HomePlanet
- Used LinearRegression to determine their room number based on their group id
- Randomly assignes the side of the aisle

#### Rest of the features

For the rest of the features I used simpler logic:

- Assign 0 for billing if the passenger was in cryosleep, otherwise median
- Age median -> then bin into decades
- Destination mode
- VIP mode (False, as VIP is meant to be exclusive)

## Statistical Inference

Reviewed multiple hypotheses that I made about the data based on the plots redarding the target variable using z-test for difference in group proportion.

## Model Building

The notebook also contains code for building a set of machine learning models to predict the target variable `Transported`.

After tuning the hyperparameters for the initialized models, the best performing models are selected for making predictions on the test data using a voting classifier.

I also added an AutoML model using the `TPOT` library to see if it can find a better model than the ones I manually created, and included it in the final ensemble a version with my preprocessed data pipeline and also a version 'let loose' on the raw data, to see if it discovers other methods and feature selection than I did by hand.

### Model Selection

I used multiple types of models, the **best performing** ones in bold:

- LogisticRegression
- KNN
- SVC
- **LightGBM**
- **Random Forest**
- **CatBoost**

I also implemented AutoML using the `TPOT` library to see if it can find a better model than the ones I manually created.

- TPOT with preprocessed data did not significantly outperform the manually created models, but I included it in the final ensemble.
- TPOT with raw data did not found I model in 300 minutes, so I stopped it.

## Results

The final ensemble model achieved an accuracy of 0.80173 on the provided test data.

## Leaderboard

Follow the leaderboard on the [Kaggle competition page](https://www.kaggle.com/competitions/spaceship-titanic/leaderboard).

## Credit

Big thank you to [Samuel Cortinhas](https://github.com/samuelcortinhas) for his extremely helpful [Kaggle notebook](https://www.kaggle.com/code/samuelcortinhas/spaceship-titanic-a-complete-guide?kernelSessionId=92521620) which was used as a reference for this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
