"""PyFlyt helper functions."""
from gymnasium.utils import colorize as gym_colorize


def colorize(string, color, bold=False, highlight=False):
    """Colorize a string.

    .. seealso::
        This function wraps the :meth:`gym.utils.colorize` function to make sure that it
        also works with empty color strings.

    Args:
        string (str): The string you want to colorize.
        color (str): The color you want to use.
        bold (bool, optional): Whether you want the text to be bold. Defaults to
            ``False``.
        highlight (bool, optional):  Whether you want to highlight the text. Defaults to
            ``False``.

    Returns:
        str: Colorized string.
    """
    if color:  # If not empty.
        return gym_colorize(string, color, bold, highlight)
    else:
        return string
