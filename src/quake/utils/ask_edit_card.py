import os
import six
import signal
from typing import Callable
from logging import Logger
from pathlib import Path
import subprocess as sp

MAX_ITERATIONS = 10  # maximmum number of allowed failing attempts


class TimeOutError(Exception):
    """Class for run-time error"""


def question_instance(logger: Logger, question: str) -> str:
    """Asks for user input and checks if the answer is valid

    Parameters
    ----------
    logger: Logger
        The logger instance.
    question: str
        The question to ask.

    Returns
    -------
    str
        The user answer to the question.
    """
    valid_results = ["0", "auto", "done", "1"]
    logger.info(question)
    counter = 0
    while True:
        result = input().lower()
        if result in valid_results:
            break
        try:
            result = Path(result)
            if result.is_file():
                break
        except:
            logger.info("Valid inputs: [0, done, 1, runcard, enter path], try again.")
        if counter >= MAX_ITERATIONS:
            break
        counter += 1
    return result


def timed_input(
    logger: Logger,
    question: str,
    default: str,
    timeout: float = None,
    noerror: bool = True,
    fct: Callable = None,
) -> str:
    """Poses a question with a maximal time to answer, take default otherwise.

    Parameters
    ----------
    logger: Logger
        The logger instance.
    question: str
        The question to ask.
    default: str
        The default answer.
    timeout: float
        Time limit to answer in seconds.
    noerror: bool
        Wether to raise error on TimeOutError exception.
    fct: Callable
        The callable effectively asking the question.

    Returns
    -------
    str
        The user answer to the question.
    """

    def handle_alarm(signum, frame):
        raise TimeOutError

    signal.signal(signal.SIGALRM, handle_alarm)

    if fct is None:
        fct = six.moves.input

    if timeout:
        signal.alarm(timeout)
        question += "[%ss to answer] " % (timeout)
    try:
        result = fct(logger, question)
    except TimeOutError:
        if noerror:
            logger.info("use %s" % default)
            return default
        else:
            signal.alarm(0)
            raise
    finally:
        signal.alarm(0)
    return result


def ask_question(
    logger: Logger, question: str, default: str, timeout: float = 10
) -> str:
    """Asks question to user, who has only `timeout` seconds to answer, then the
    `default` is returned.

    Parameters
    ----------
    logger: Logger
        The logger instance.
    question: str
        The question to ask.
    default: str
        The default answer.
    timeout: float
        Time limit to answer in seconds.

    Returns
    -------
    str
        The user answer to the question.
    """
    value = timed_input(
        logger, question, default, timeout=timeout, fct=question_instance
    )
    return value


def ask_edit_card(logger: Logger, output: Path):
    """Asks interactively to edit the runcard.

    Receives the input from the user and opens an editor in the terminal as a
    subprocess. Default editor is nano, otherwise the `QUAKE_EDITOR` environment
    variable allows for custom choice.

    Parameters
    ----------
    logger: logger
        The logger instance.
    output: Path
        The output folder.
    """
    default_quake_editor = "nano"
    editor = os.environ.get("QUAKE_EDITOR", default_quake_editor)

    question = """Do you want to edit a card (press enter to bypass editing)?
Type '0', 'auto', 'done' or just press enter when you are done
[0, done, 1, runcard, enter path]"""

    answer = ask_question(logger, question, "0", timeout=60)

    if answer not in ["0", "auto", "done"]:
        if answer == "1":
            fname = output / "cards/runcard.yaml"
        else:
            fname = answer
        sp.call([editor, fname])
        logger.info(f"Updated runcard at {fname}")
