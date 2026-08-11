import sys

print("1. Importing torch...")
import torch
print("   Torch imported successfully.")

print("2. Importing matplotlib...")
import matplotlib
print("   Matplotlib imported successfully.")

print("3. Setting matplotlib backend to Agg...")
matplotlib.use("Agg")
print("   Matplotlib backend set successfully.")

print("4. Importing pyplot...")
import matplotlib.pyplot as plt
print("   Pyplot imported successfully.")

print("5. Importing seaborn...")
import seaborn as sns
print("   Seaborn imported successfully.")

print("6. Importing scikit-learn...")
import sklearn
print("   Scikit-learn imported successfully.")

print("All imports completed successfully!")
