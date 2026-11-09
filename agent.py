"""Compatibility launcher; implementation lives in gamelens.agents.pipeline."""
if __name__ == "__main__":
    from gamelens.bootstrap import ensure_environment
    ensure_environment()

from gamelens.agents.pipeline import main

if __name__ == "__main__":
    main()
