"""
This module defines the PokerAgent class responsible for decision-making in a Poker-playing agent.

The PokerAgent interacts with the game state and implements the strategy for determining the next action in the game. 
It includes various methods for handling game events such as the start and end of a game or round, image recognition 
(for cards), and error handling.

Dependencies:
- 'model': Contains methods like `load_model()` and `identify()`, which are used by the PokerAgent.
- 'client.state': Contains `ClientGameRoundState` and `ClientGameState`, which represent the current game and round states.
"""
import random
from client.controller import Controller
from model import load_model, identify
from client.state import ClientGameRoundState, ClientGameState
from rl_model import *
from sklearn.preprocessing import OneHotEncoder

class PokerAgent(object):

    def __init__(self, rl_agent=None, num_cards=3): # Adjust num_cards based on the type of game
        self.model = load_model() # Loads the saved model for image recognition
        self.actions = ['BET', 'CALL', 'CHECK', 'FOLD']
        self.agent_state = None  # State composed by the current card and opponent action
        self.state_size = num_cards+3  # Assume 6 or 7 input features (3 or 4 for card + 3 for opponent action)
        self.chosen_action_index = None
        self.rl_agent = rl_agent or RLAgent(self.state_size, len(self.actions)) # Use the provided rl_agent instance
        self.current_card_rank = None
        self.game_type = num_cards - 3  # Is '0' if the game is played with 3 cards '1' if it is played with 4 cards

        
        # One-hot encoder for cards and opponent actions
        card_categories = ['J', 'Q', 'K', 'A'][:num_cards] # Adjust categories based on num_cards
        self.encoder = OneHotEncoder(sparse_output=False, categories=[card_categories, ['BET', 'CHECK', None]]) # Handle 3 or 4 cards encoding

    def make_action(self, state: ClientGameState, round: ClientGameRoundState) -> str:
        """
        Choose the next action depending on the current state of the game. This method implements the PokerBot strategy.

        Parameters
        ----------
        state : ClientGameState
            State object of the current game (a game has multiple rounds)
        round : ClientGameRoundState
            State object of the current round (from deal to showdown)

        Returns
        -------
        str in ['BET', 'CALL', 'CHECK', 'FOLD'] (and in round.get_available_actions())
            The next action based on available actions and the RL agent's decision.
        """

        available_actions = round.get_available_actions()

        # Encode the current state (card + opponent action)
        self.agent_state = self.encode_state(round)

        # Convert available actions to indices
        available_action_indices = self.get_available_action_indices(available_actions)

        # Choose the best action from the RL agent based on the state and the available actions
        self.chosen_action_index = self.rl_agent.choose_action(self.agent_state, available_action_indices)

        # Map back the chosen action index to its string representation
        return self.actions[self.chosen_action_index]

    def encode_state(self, round: ClientGameRoundState):
        """
        Encodes the card and opponent's action as a one-hot vector.

        Parameters
        ----------
        round : ClientGameRoundState
            State object of the current round (from deal to showdown)

        Returns
        -------
        np.array
            Encoded state as a one-hot vector for the card and opponent's action.
        """

        # Get the last action from the opponent
        opponent_action = round.get_moves_history()[-1] if round.get_moves_history() else None

        # Create the state by joining the card rank and the opponent action
        state = [[self.current_card_rank, opponent_action]]

        # Compute the one-hot vector of the state
        encoded_state = self.encoder.fit_transform(state).flatten()

        # Return the agent state in a tensor
        return tf.convert_to_tensor([encoded_state], dtype=tf.float32)
        
    def get_available_action_indices(self, available_actions: list) -> list:
        """
        Converts available actions into their corresponding indices.

        This method maps the available actions to their respective index in the predefined list of actions.

        Parameters
        ----------
        available_actions : list of str
        List of available actions in the current round (e.g., ['BET', 'CHECK']).

        Returns
        -------
        list of int
        A list of indices corresponding to the available actions in the agent's action space.
        """
        return [self.actions.index(action) for action in available_actions]

    def on_image(self, image):
        """
        This method is called every time when the card image changes. Use this method for image recongition.

        Parameters
        ----------
        image : Image
            Image object
        """
        # Simulated identification of card rank from the image (to be replaced with actual image recognition logic)
        # self.current_card_rank = random.choice(['J', 'Q', 'K'])

        self.current_card_rank = identify(image, self.model, self.game_type) 

    def on_error(self, error):
        """
        This methods will be called in case of error either from the server backend or from the client itself.

        Parameters
        ----------
        error : str
            String representation of the error
        """
        print(f"Error: {error}")

    def on_game_start(self):
        """
        This method will be called once at the beginning of the game when the server has confirmed that both players are connected.
        """
        pass

    def on_new_round_request(self, state: ClientGameState):
        """
        This method is called every time before a new round is started. A new round is started automatically.

        Parameters
        ----------
        state : ClientGameState
            State object of the current game
        """
        print("New round is starting!")

    def on_round_end(self, state: ClientGameState, round: ClientGameRoundState):
        """
        This method is called every time a round has ended. A round ends automatically. 

        Parameters
        ----------
        state : ClientGameState
            State object of the current game
        round : ClientGameRoundState
            State object of the current round
        """
        round.set_card(self.current_card_rank)
        print(f'----- Round {round.get_round_id()} results -----')
        print(f'  Recognized card : {round.get_card()}')
        print(f'  Your turn order : {round.get_turn_order()}')
        print(f'  Moves history   : {round.get_moves_history()}')
        print(f'  Your outcome    : {round.get_outcome()}')
        print(f'  Current bank    : {state.get_player_bank()}')
        print(f'  Show-down       : {round.get_cards()}')

        if self.rl_agent.training:
            # Update the Q-network if training is activated
            reward = round.get_outcome()
            self.rl_agent.update(self.chosen_action_index, self.agent_state, reward)
        
    def on_game_end(self, state: ClientGameState, result: str):
        """
        Called once after the game has ended automatically.

        Parameters
        ----------
        state : ClientGameState
            State object of the current game
        result : str in ['WIN', 'DEFEAT']
            End result of the game
        """
        print(f'----- Game results -----')
        print(f'  Outcome:    {result}')
        print(f'  Final bank: {state.get_player_bank()}')

        if self.rl_agent.evaluating and result == 'WIN':
            self.rl_agent.wins += 1

def train(rl_agent : RLAgent, num_games=10):
    """
    Train the RL agent by playing a specified number of games against a bot.

    This function initializes a connection to the poker game server, sets the RL agent into training mode, 
    and updates its exploration rate after each game. The RL agent is trained by playing multiple games,
    and the loss for each game is recorded and stored in the agent's loss history.

    Parameters
    ----------
    rl_agent : RLAgent
        The RL agent responsible for making decisions during gameplay and updating based on learning.
    
    num_games : int, optional
        The number of games the agent will play during training. Default is 10.

    Returns
    -------
    None
    """
    # server_address = '51.159.25.188:50051'  # Address of the poker server (can switch to localhost for testing)
    server_address = 'localhost:50051'  # Address of the poker server

    # Authentication token to connect to the poker game server
    try:
            with open("token_key.txt", "r") as f:
                token = f.read(36)
    except FileNotFoundError:
            print('Token has not been specified. Create a `token_key.txt`.')
            return
    except Exception:
            print('Error reading token from `token_key.txt` file. Ensure that token has a valid UUID structure and has not extra spaces before and after the token.')
            return
    
    client = Controller(token, server_address)  # Create a client to interact with the server
    
    rl_agent.epsilon = 1  # Start with full exploration (epsilon = 1)
    rl_agent.training = True  # Set the agent to training mode
    
    for game in range(num_games):
        rl_agent.loss_history = []  # Clear the loss history for each game
        # Play a game against the bot with the RL agent, specifying the number of cards and the agent creation function
        client.play("bot", rl_agent.num_cards, lambda: PokerAgent(rl_agent))
        
        # Decrease epsilon (exploration rate) after each game using epsilon_decay
        rl_agent.epsilon = rl_agent.epsilon_decay * rl_agent.epsilon
        
        # Use the average because games last for different rounds
        average_loss = sum(rl_agent.loss_history) / len(rl_agent.loss_history)

        # Append the average loss for the game to the agent's game loss history
        rl_agent.game_loss.append(average_loss)
    
    rl_agent.epsilon = 0  # After training, set epsilon to 0 (no exploration)
    rl_agent.training = False  # Disable training mode after all games are played

def evaluate(rl_agent : RLAgent, num_games=10):
    """
    Evaluate the performance of a pre-trained RL agent by playing a specified number of games against a bot.

    Parameters
    ----------
    rl_agent : RLAgent
        The RL agent responsible for making decisions during gameplay and updating based on learning.

    num_games : int, optional
        The number of games to play during evaluation (default is 10).

    Returns
    -------
    float
        The win rate percentage after playing the specified number of games.
    """
    # server_address = '51.159.25.188:50051'  # Address of the poker server (can switch to localhost for testing)
    server_address = 'localhost:50051'  # Address of the poker server

    # Authentication token to connect to the poker game server
    try:
            with open("token_key.txt", "r") as f:
                token = f.read(36)
    except FileNotFoundError:
            print('Token has not been specified. Create a `token_key.txt`.')
            return
    except Exception:
            print('Error reading token from `token_key.txt` file. Ensure that token has a valid UUID structure and has not extra spaces before and after the token.')
            return
    client = Controller(token, server_address)
    rl_agent.wins = 0
    rl_agent.evaluating = True
    for game in range(num_games):
        client.play("bot", rl_agent.num_cards, lambda: PokerAgent(rl_agent))
    rl_agent.evaluating = False
    return  rl_agent.wins/num_games