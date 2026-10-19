"""Re-evaluate the saved checkpoint using the exact original run arguments."""
import sys
from train import main

if __name__ == '__main__':
    if '--evaluate-only' not in sys.argv:
        sys.argv.append('--evaluate-only')
    main()
