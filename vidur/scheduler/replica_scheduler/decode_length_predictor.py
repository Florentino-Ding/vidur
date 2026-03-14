import numpy as np

from vidur.entities import Request


class DecodeLengthPredictor:
    """
    Predicts decode length for a request by sampling from N(true_length, noise_std**2),
    clamped to >= 1.  noise_std is expressed in tokens.
    """

    def __init__(self, noise_std: float = 0.0) -> None:
        self._noise_std = noise_std

    def predict(self, request: Request) -> float:
        true_length = float(request.num_decode_tokens)
        return max(1.0, float(np.random.normal(true_length, self._noise_std)))
