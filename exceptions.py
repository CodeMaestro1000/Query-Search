class NoCrossEncoder(Exception):
    """For throwing when trying to use cross encoder that wasn't specified"""

class NoDataAvailable(Exception):
    """For throwing when trying to use train a model where the data wasn't specified"""