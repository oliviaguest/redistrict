"""Get data from OSF repository."""
from __future__ import print_function
import osfclient.cli as cli

# https://osfclient.readthedocs.io/en/stable/index.html


class Args():
    """I am  not ashamed of this hack."""

    def __init__(self,
                 remote,
                 #  local,
                 username,
                 project,
                 output):
        """Or am I? ...OK, still not ashamed."""
        self.remote = remote
        # self.local = local
        self.username = username
        self.project = project
        self.output = output


args = Args(
    'osfstorage/',
    # './temp/',
    'guest.olivia@gmail.com',
    '8xpq4',
    '.')
cli.clone(args)
