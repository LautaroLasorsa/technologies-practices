"""Allow running the producer with: uv run python -m app.producer

Or running the faust worker with: uv run faust -A app.main worker -l info
"""

from app.producer import main

if __name__ == "__main__":
    main()
