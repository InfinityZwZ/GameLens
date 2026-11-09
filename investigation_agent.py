"""Compatibility launcher; implementation lives in gamelens.agents.investigation."""
if __name__ == "__main__":
    from gamelens.bootstrap import ensure_environment
    ensure_environment()

from gamelens.agents.investigation import main

if __name__ == "__main__":
    main()
