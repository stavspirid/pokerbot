import unittest
from unittest.mock import Mock
import numpy as np
import tensorflow as tf
from agent import PokerAgent


class TestPokerAgent(unittest.TestCase):

    def setUp(self):
        # Mock RLAgent
        self.mock_rl_agent = Mock()
        self.mock_rl_agent.choose_action.return_value = 0  # Mock 'BET' as the action (index 0)
        self.mock_rl_agent.update = Mock()

        # Initialize PokerAgent with the mock RLAgent
        self.poker_agent = PokerAgent(rl_agent=self.mock_rl_agent, num_cards=3)

        # Mock game state and round state
        self.mock_game_state = Mock()
        self.mock_round_state = Mock()
        self.mock_round_state.get_available_actions.return_value = ['BET', 'CHECK']
        self.mock_round_state.get_moves_history.return_value = ['CHECK']

    def test_make_action(self):
        # Test the make_action method
        self.poker_agent.encode_state = Mock(return_value=tf.convert_to_tensor([[0, 1, 0]], dtype=tf.float32))

        action = self.poker_agent.make_action(self.mock_game_state, self.mock_round_state)

        # Ensure the right action is chosen (mock returns index 0, which is 'BET')
        self.assertEqual(action, 'BET')

    def test_encode_state(self):
        # Set the current card rank and mock the moves history
        self.poker_agent.current_card_rank = 'Q'
        self.mock_round_state.get_moves_history.return_value = ['CHECK']

        # Test encode_state (no need for advanced mock; we just check the behavior)
        encoded_state = self.poker_agent.encode_state(self.mock_round_state)

        # Check that the encoded state is a tensor
        self.assertIsInstance(encoded_state, tf.Tensor)

    def test_get_available_action_indices(self):
        # Test converting available actions to indices
        available_actions = ['BET', 'CHECK']
        action_indices = self.poker_agent.get_available_action_indices(available_actions)

        # Assert that 'BET' is index 0 and 'CHECK' is index 2
        self.assertEqual(action_indices, [0, 2])

    def test_on_image(self):
        # Mock the identify function to return a specific card rank
        self.poker_agent.on_image = Mock(return_value='Q')
        mock_image = Mock()

        # Call on_image and check if it sets the correct card rank
        self.poker_agent.on_image(mock_image)
        self.assertEqual(self.poker_agent.current_card_rank, 'Q')

    def test_on_round_end(self):
        # Mock outcome, card rank, and update
        self.mock_round_state.get_round_id.return_value = 1
        self.mock_round_state.get_card.return_value = 'Q'
        self.mock_round_state.get_outcome.return_value = 1  # Win outcome
        self.mock_game_state.get_player_bank.return_value = 1000

        # Call on_round_end and check if update method of RLAgent is called
        self.poker_agent.on_round_end(self.mock_game_state, self.mock_round_state)
        self.mock_rl_agent.update.assert_called_once()

    def test_on_game_end(self):
        # Mock the game state and result
        self.mock_game_state.get_player_bank.return_value = 1500

        # Call on_game_end with result 'WIN'
        self.poker_agent.on_game_end(self.mock_game_state, 'WIN')

        # Check that the game result was printed and wins are updated
        self.assertEqual(self.mock_rl_agent.wins, 1)


if __name__ == '__main__':
    unittest.main()