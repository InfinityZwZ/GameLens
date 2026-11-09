"""Compatibility launcher; implementation lives in gamelens.evaluation.cli."""
if __name__ == "__main__":
    from gamelens.bootstrap import ensure_environment
    ensure_environment()

from gamelens.evaluation.cli import main

if __name__ == "__main__":
    main()
