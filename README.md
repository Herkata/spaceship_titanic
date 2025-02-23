# Spaceship Titanic Kaggle Competition

Welcome to the Spaceship Titanic Kaggle Competition! This repository contains code and resources for participating in the competition.

## Overview

The goal of this competition is to predict which passengers were transported to an alternate dimension during the collision of the Spaceship Titanic. 

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
- `helper.py`: Additional code for data processing and visualization.
- `requirements.txt`: List of required Python libraries.

## Citation

@misc{spaceshiptitanic,
    title={Spaceship Titanic Kaggle Competition},
    url={https://www.kaggle.com/competitions/spaceship-titanic/overview/$citation},
    note={Accessed: [Insert Date]}
}

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
