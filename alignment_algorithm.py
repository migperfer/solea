# import librosa
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from numpy.typing import NDArray
from typing import Optional, Tuple


def get_events_alignment(
        audio_events: NDArray,
        score_events: NDArray,
        chroma_warping_path: NDArray,
        threshold: float,
        audio_class: Optional[NDArray] = None,
        midi_class: Optional[NDArray] = None) -> Tuple[NDArray | None, NDArray | None, NDArray | None]:
    """
    The event-based alignment algorithm
    :param audio_events: A numpy array of size N containing the timestamps of the audio events
    :param score_events: A numpy array of size M containing the timestamps of the score events
    :param chroma_warping_path: A numpy array containing the warping path from the audio to the score
    :param threshold: The maximum divergence allowed to align a pair of audio/score events
    :param audio_class: A numpy array of size N containing the class types associated to the audio events
    :param midi_class: A numpy array of size M containing the class types associated to the score events
    :return: Returns a tuple of three arrays. The first element is the new warping path. The second and third elements
    are the aligned events in the score and the audio respectively.
    """

    audio_len = len(audio_events)

    wp_interpolator = scipy.interpolate.interp1d(
        x=chroma_warping_path[:, 0],
        y=chroma_warping_path[:, 1],
        kind="linear",
        axis=0,
        bounds_error=False,
        fill_value="extrapolate",
    )

    time_interpolated_from_audio_events = wp_interpolator(audio_events)
    cost_matrix = abs(time_interpolated_from_audio_events[:, np.newaxis] - score_events[np.newaxis, :].repeat(audio_len, 0))

    epsilon = 1e-6
    cost_matrix = np.clip(cost_matrix, epsilon, a_max=np.inf)
    cost_matrix[cost_matrix > threshold] = 0
    if (audio_class is not None) and (midi_class is not None):
        for i, a_class in enumerate(audio_class):
            for j, midi_class in enumerate(midi_class):
                cost_matrix[i, j] = 0

    cost_matrix = csr_matrix(cost_matrix)
    matches = maximum_bipartite_matching(cost_matrix, perm_type="column")
    audio_onset_idxs = np.where(matches > -1)[0]
    score_onset_idxs = matches[audio_onset_idxs]

    if not all(np.diff(score_onset_idxs) > 0):
        return None, None, None  # There should be a better way of ensuring that the onsets are monotonically increasing

    new_wp = np.zeros((len(audio_onset_idxs), 2))
    new_wp[:, 0] = audio_events[audio_onset_idxs]
    new_wp[:, 1] = score_events[score_onset_idxs]
    return new_wp, score_onset_idxs, audio_onset_idxs
