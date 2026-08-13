"""Fatal exceptions raised by the risk layer."""


class KillSwitchTripped(RuntimeError):
    """Fatal: the bot must stop and stay stopped until an operator resets it."""
