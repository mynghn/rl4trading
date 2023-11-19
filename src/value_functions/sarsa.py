import numpy as np

from value_functions.interface import ValueFunction


class LinearSarsa(ValueFunction):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        learning_rate: float,
        discount_factor: float,
    ):
        self.params = np.ones(dim_state + dim_action) * 100
        self.alpha = learning_rate
        self.gamma = discount_factor

    def predict(self, state: np.ndarray, action: np.ndarray) -> float:
        state_action_vector = np.concatenate((state, action))
        return np.dot(self.params, state_action_vector)

    def argmax(self, state: np.ndarray) -> np.ndarray:
        action_params = self.params[len(state) :]

        argmax_action = np.zeros(len(action_params))
        argmax_action[np.argmax(action_params)] = 1

        return argmax_action

    def loss_function(
        self,
        reward: float,
        curr_state: np.ndarray,
        curr_action: np.ndarray,
        next_state: np.ndarray,
    ) -> float:
        actual = reward + self.predict(
            state=next_state, action=self.argmax(state=next_state)
        )
        pred = self.predict(state=curr_state, action=curr_action)

        return actual - pred

    def update(
        self,
        reward: float,
        curr_state: np.ndarray,
        curr_action: np.ndarray,
        next_state: np.ndarray,
    ):
        loss = self.loss_function(
            reward=reward,
            curr_state=curr_state,
            curr_action=curr_action,
            next_state=next_state,
        )
        delta = self.alpha * loss * np.concatenate((curr_state, curr_action))

        self.params += delta
