"""Compatibility launcher; implementation lives in gamelens.evaluation.benchmark."""
if __name__ == "__main__":
    from gamelens.bootstrap import ensure_environment
    ensure_environment()

def main():
    import sys
    from gamelens.evaluation.benchmark_v1 import COMMANDS, main as unified_main
    if len(sys.argv)>1 and sys.argv[1] in COMMANDS:
        return unified_main()
    from gamelens.evaluation.benchmark import main as legacy_main
    return legacy_main()


if __name__ == "__main__":
    raise SystemExit(main())
