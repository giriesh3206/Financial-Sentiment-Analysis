import sys
import os
import traceback

sys.path.append(os.path.abspath("."))

try:
    print("Attempting to import src.training.train_rnn_lstm...")
    import src.training.train_rnn_lstm
    print("Import successful!")
except Exception as e:
    print("Import failed with exception:")
    traceback.print_exc()
