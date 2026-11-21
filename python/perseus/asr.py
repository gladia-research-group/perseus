"""`python -m perseus.asr`: encrypted speech-to-text from the command line (perseus.examples.whisper.cli; --help)."""
import sys

from perseus.examples.whisper.cli import main

if __name__ == "__main__":
    sys.exit(main())
