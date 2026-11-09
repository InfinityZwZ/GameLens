"""Local evaluation launcher. Offline by default; live inference requires opt-in."""
if __name__ == "__main__":
    from gamelens.bootstrap import ensure_environment
    ensure_environment()

from gamelens.evaluation.suite import main

if __name__ == "__main__":
    raise SystemExit(main())
