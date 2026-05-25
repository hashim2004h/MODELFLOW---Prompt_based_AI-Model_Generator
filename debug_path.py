import os
import sys

print(f"Python Executable: {sys.executable}")
print(f"CWD: {os.getcwd()}")
try:
    print(f"Files in CWD: {os.listdir('.')}")
except Exception as e:
    print(f"Error listing CWD: {e}")
