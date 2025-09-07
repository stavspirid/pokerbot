# Documentation

## Overview

This project implements a poker bot agent capable of playing a 3-card and 4-card Kuhn poker game. The agent utilizes a image recognizer to detect the dealt cards, allowing it to make informed decisions during gameplay.

## Features

- **Card Image Recognition**: Automatically detects and classifies dealt cards using a trained image recognition model.
It generates a prediction vector with all cards ranked from most to least probable. Then, the model returns the most probable card using the `identify` method. The model can be found on `model.py` file.
- **Kuhn Poker Strategy**: Implements the strategies for playing Kuhn poker. The whole game strategy can be found on the `agent.py` file.

## Installation

To set up the project, follow the instructions in the Setup section in the [README.md](https://github.com/tue-5ARA0-2024-Q1/pokerbot-pokerbot-10/blob/main/README.md) file.

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Create a new branch (Topic branching)
2. Make your changes and commit them
3. Push to the branch
4. Open a pull request

### Contribution Guidelines
Please keep the following guidelines in mind:

1. Create an issue if you encounter a problem or want to propose a feature.
2. Follow SOLID principles and use AGILE practices.
3. Write clear Docstrings and comments to explain your code.
4. Implement data and model versioning to track changes.
5. Use the Kanban board in the Projects tab to manage the workflow.
6. Write unit tests when adding new features or making significant changes.

## Design choices

### Selected Machine Learning Algorithms
For the **Image Recognition Model** a *Convolutional Neural Network* model was used to classify playing cards by rank (Jack, Queen, King, Ace). The CNN's architecture uses:
- Three convolutional layers with ReLU activations and MaxPooling for feature extraction.
- A Flatten layer followed by a Dense layer with 256 neurons.
- A Dropout layer (50%) to prevent overfitting.
- An output Dense layer with four neurons and softmax activation to classify the four card ranks

For the **Poker Agent Model** a *Reinforcement Learning* model was used with this characteristics:
- Exploration Rate (Epsilon): The agent uses an epsilon-greedy strategy, starting with a high exploration rate (epsilon = 1, meaning full exploration at the start).
- Epsilon gradually decreases as the agent learns, controlled by epsilon_decay.

### 3/4 Card Games
Because the performance of the Image Recognition model is more than sufficient, we decided to not write 2 different models for each type of game (3 or 4 card games).

So an extra step was implemented on the model that in case of an unexpeted wrong recognition of the card (unexptected due to high confidence on our testing) the model returns only cards that are suitable to the game type.

This means that if the game type is a 3-cards Kuhn Poker game and the model returns the `Ace` card, we switch this card with the next most probable card.

## Training
### Downloading Training Datashet
To download the image training dataset use [this](https://drive.google.com/drive/folders/1SdNWVRyAc1YKp1W0KlOryRSOMlaKW-m6) *Google Drive* link and download `dataset_train_4_15000_08.npz`. 
Put the downloaded file in the `dataset/` folder.

### Training the Image Recognition Model
To train the CNN model use the commands provided on the `README.md` file that are listed here too for ease of use.

Use the `--build-model` flag to build an image recognition model:
```
> python main.py --build-model
```

Use the `--train-model 'n_validation'` command to train image recognition model with the specified `n_validation`:
```
> python main.py --train-model 100
```

### Training the Agent
You can find all the information about training the agent on the `train_agent.ipynb` file.

## Gamepley Video
A game play video can be found on the Canvas submission.

## Known Bugs and Mistakes
The 4 rank agent `rl_model_4_card_eps-98_games250.pkl` had been trained with 3 rank image recognition, so it could never recognize `'A' (Ace)`. Unfortunately this mistake have been recognized too late into the development, and training a corrected model was not possible.

A solution to this problem is very easy to do but due to the late recognission of the problem it was not possible. We believe that if the model was trained on the correct image recognition file, it would be as affective as the 3-card model.