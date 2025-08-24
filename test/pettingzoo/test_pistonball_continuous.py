import argparse

import pytest
from pistonball_continuous import get_args, train_agent, watch


@pytest.mark.slow  # Moved to slow: Runtime too long and unstable (see test/ICEBOX_TESTS.md)
def test_piston_ball_continuous(args: argparse.Namespace = get_args()) -> None:
    if args.watch:
        watch(args)
        return

    result, agent = train_agent(args)
    # assert result.best_reward >= 30.0
