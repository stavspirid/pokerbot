import os
import joblib
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

RL_MODEL_PATH = os.path.join(ROOT_DIR, "rl_models")

class QNetwork(tf.keras.Model):
    """
    A neural network model for approximating Q-values in Reinforcement Learning.
    
    This class inherits from tf.keras.Model and represents a simple feedforward neural network
    with two hidden layers and one output layer. The output layer provides Q-values for all possible actions.
    """
    def __init__(self, input_size, output_size):
        """
        Initialize the QNetwork model.

        Parameters
        ----------
        input_size : int
            Number of input features.
        output_size : int
            Number of output neurons, corresponding to the number of possible actions.
        """
        super(QNetwork, self).__init__()

        # First layer
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')

        #Second layer
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')

        # Output layer
        self.fc3 = tf.keras.layers.Dense(output_size, activation=None)  

    def call(self, inputs):
        """
        Forward pass through the network.

        Parameters
        ----------
        inputs : tf.Tensor
            A batch of input states represented as a tensor.

        Returns
        -------
        tf.Tensor
            A tensor of predicted Q-values for each possible action.
        """
        # Pass inputs through the first layer
        x = self.fc1(inputs)

         # Pass the result through the second layer
        x = self.fc2(x)

        # Return the output, which represents the Q-values for each action
        return self.fc3(x)
    
def save_rl_model(rl_model, num_cards=3, rl_model_path=RL_MODEL_PATH):
    """
    Saves the RL model to a file. The saved model is named based on 
    the number of cards used in the game (e.g., 'rl_model_3_card.pkl' for a 3-card game).

    Parameters
    ----------
    rl_model : object
        The reinforcement learning model object to be saved.
    num_cards : int, optional
        The number of cards in the game, either 3 or 4. This determines the file name (default is 3).
    rl_model_path : str, optional
        The directory path where the RL model file will be saved. This is defaulted to RL_MODEL_PATH.
    
    Returns
    -------
    rl_model : object
        The input RL model object is returned, allowing for further usage after saving.
    """
    # Create the model filename based on the number of cards
    rl_model_name = f"rl_model_{num_cards}_card.pkl"

    # Ensure that the directory for saving the RL model exists
    os.makedirs(rl_model_path, exist_ok=True)

    # Create the full path to the RL model file
    rl_model_file = os.path.join(rl_model_path, rl_model_name)

    # Save the RL mode
    joblib.dump(rl_model, rl_model_file)

    return rl_model

def load_rl_model(num_cards=3, rl_model_path=RL_MODEL_PATH):
    """
    Loads a RL model from a file. The model is loaded based on the number of 
    cards used in the game (e.g., 'rl_model_3_card.pkl' for a 3-card game).

    Parameters
    ----------
    num_cards : int, optional
        The number of cards in the game, either 3 or 4. This determines the file name (default is 3).
    rl_model_path : str, optional
        The directory path where the RL model file is stored. This is defaulted to RL_MODEL_PATH.

    Returns
    -------
    rl_model : object
        The loaded RL model object that can be used for decision making.
    """
    # Create the model filename based on the number of cards
    rl_model_name = f"rl_model_{num_cards}_card.pkl"

    # Create the full path to the RL model file
    rl_model_file = os.path.join(rl_model_path, rl_model_name)
    
    return joblib.load(rl_model_file)

class RLAgent:
    """
    Reinforcement Learning Agent for playing Kuhn poker. 
    This agent can be trained and evaluated to make optimal decisions based on the Q-learning algorithm.

    Attributes
    ----------
    rl_model : QNetwork
        The neural network model used to approximate the Q-values for state-action pairs.
    optimizer : tf.keras.optimizers.Optimizer
        The optimizer used to update the weights of the model during training.
    loss_fn : tf.keras.losses.Loss
        The loss function used to calculate the difference between the predicted and target Q-values.
    loss_history : list
        A list that stores the loss values after every round of the last game.
    game_loss : list
        A list that stores the total loss values after every complete game.
    epsilon : float
        The exploration rate used for epsilon-greedy action selection.
    epsilon_decay : float
        The rate at which epsilon decreases over time.
    training : bool
        Flag that indicates whether the agent is in training mode.
    evaluating : bool
        Flag that indicates whether the agent is in evaluation mode (no learning).
    wins : int
        The number of games won by the agent when being evaluated.
    num_cards : int
        Number of different cards in the game
    """

    def __init__(self, state_size=6, action_size=4, rl_model = None, learning_rate=0.001):
        self.rl_model = rl_model or QNetwork(state_size, action_size) # Initialize the reinforcement learning model (Q-Network)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) # Optimizer used for updating the RL model parameters
        self.loss_fn = tf.keras.losses.MeanSquaredError() #Type of loss function
        self.loss_history = []  # Initialize an array to store loss values for each round of the last game
        self.game_loss = []  # Initialize an array to store the total loss values for each game
        self.epsilon = 0  # Not exploring, only if training
        self.epsilon_decay = 0.75 # Decay rate of epsilon
        self.training = False # Flag for training
        self.evaluating = False # Flag for evaluating
        self.wins = 0 # Number of games won when being evaluated
        self.num_cards = state_size-3 # Number of different cards in the game
        
    def choose_action(self, agent_state, available_action_indices):
        """
        Choose an action based on the epsilon-greedy strategy.
        
        The epsilon-greedy strategy balances exploration and exploitation. With a probability
        of epsilon, the agent will explore (choose a random action). Otherwise, it will
        exploit its current knowledge (choose the action with the highest Q-value).

        Parameters
        ----------
        agent_state : np.array
            The current state of the agent (a feature vector representing the game state).
        available_action_indices : list of int
            A list of action indices that are available to the agent in the current game round.
        epsilon : float
            The exploration rate (epsilon).

        Returns
        -------
        int
            The index of the chosen action based on the epsilon-greedy strategy.
        """

        if np.random.rand() < self.epsilon:
            # Exploration: Choose a random valid action
            return np.random.choice(available_action_indices)
        
        # Exploitation: Choose the action with the highest Q-value among available actions
        
        # Get Q-values
        q_values = self.rl_model(agent_state).numpy()[0]
    
        # Only consider available actions
        available_q_values = q_values[available_action_indices]

        # Choose the best action
        best_action_index = np.argmax(available_q_values)

        # Return the index of the best action
        return available_action_indices[best_action_index]

    def update(self, action_index, agent_state, reward):
        """
        Perform a Q-learning update step for the current state-action pair.

        This method updates the Q-value for the chosen action in the current state by adjusting the weights of the 
        neural network using backpropagation.

        Parameters
        ----------
        action_index : int
            The index of the action that was taken by the agent.
        reward : float
            The reward received after taking the action.

        Returns
        -------
        None
        """
        with tf.GradientTape() as tape:
            # Get Q-values
            q_values = self.rl_model(agent_state)
            
            target_q_value = q_values.numpy().copy()  # Copy the current Q-values
            
            target_q_value[0, action_index] = reward
            target_q_value = tf.convert_to_tensor(reward, dtype=tf.float32)

            # Compute the loss (Mean Squared Error between current Q-value and target Q-value)
            loss = self.loss_fn(target_q_value, q_values)

            #If the agent is in training mode, store the loss in the loss history
            if self.training:
                self.loss_history.append(loss)
        
        # Compute the gradients of the loss with respect to the model parameters
        gradients = tape.gradient(loss, self.rl_model.trainable_variables)

        # Apply the gradients to update the weights of the neural network
        self.optimizer.apply_gradients(zip(gradients, self.rl_model.trainable_variables))

    def plot_learning_curve(self):
        """
        Plots the learning curve, showing how the loss changes over time.

        Parameters
        ----------
        losses : list
            A list of loss values recorded during training.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.game_loss, label='Training Loss')
        plt.title('Learning Curve')
        plt.xlabel('Games')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()