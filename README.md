# Spaceship Titanic Kaggle Competition Submission

This repository contains code and resources to showcase my submission to the above mentioned competition.

## Overview

The goal of this competition is to predict which passengers were transported to an alternate dimension during the collision of the Spaceship Titanic.

Access the competition page [here](https://www.kaggle.com/competitions/spaceship-titanic/overview).

## Dataset

- **PassengerId**: A unique Id for each passenger. Each Id takes the form `gggg_pp` where `gggg` indicates a group the passenger is travelling with and `pp` is their number within the group. People in a group are often family members, but not always.
- **HomePlanet**: The planet the passenger departed from, typically their planet of permanent residence.
- **CryoSleep**: Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
- **Cabin**: The cabin number where the passenger is staying. Takes the form `deck/num/side`, where side can be either `P` for Port or `S` for Starboard.
- **Destination**: The planet the passenger will be debarking to.
- **Age**: The age of the passenger.
- **VIP**: Whether the passenger has paid for special VIP service during the voyage.
- **RoomService, FoodCourt, ShoppingMall, Spa, VRDeck**: Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
- **Name**: The first and last names of the passenger.
- **Transported**: Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

## Repository Structure

- `spaceship_titanic_data/`: Contains the dataset files.
- `space_titanic.ipynb`: Jupyter notebook for data exploration and model building.
- `helper.py`: Additional code moved from the notebook for better readability.
- `requirements.txt`: List of required Python libraries.

## EDA

The notebook `spaceship_titanic.ipynb` contains exploratory data analysis and visualization of the dataset from different perspectives.

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

## Model Building

The notebook also contains code for building a set of machine learning models to predict the target variable `Transported`.

After tuning the hyperparameters for the initialized models, the best performing models are selected for making predictions on the test data using a voting classifier.

I also added an AutoML model using the `TPOT` library to see if it can find a better model than the ones I manually created, and included it in the final ensemble a version with my preprocessed data pipeline and also a version 'let loose' on the raw data, to see if it discovers other methods and feature selection than I did by hand.

### Model Selection

I used multiple types of models, the best performing ones were:
- LightGBM
- Random Forest
- SVC

## Results

The final ensemble model achieved an accuracy of 0.80173 on the provided test data.

## Leaderboard

Follow the leaderboard on the [Kaggle competition page](https://www.kaggle.com/competitions/spaceship-titanic/leaderboard).

## Credit

Big thank you to [Samuel Cortinhas](https://github.com/samuelcortinhas) for his extremely helpful [Kaggle notebook](https://www.kaggle.com/code/samuelcortinhas/spaceship-titanic-a-complete-guide?kernelSessionId=92521620) which was used as a reference for this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
