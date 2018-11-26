import sys
import os
import subprocess

a = subprocess.Popen(['/usr/bin/python3', './test.py', '1', '2', '3'])

print("this is the return value" + a)