import os
import sys


# Get the path to the 'ALIKE' directory within 'thirdparty'
thirdparty_path = os.path.join(os.path.dirname(__file__), 'ALIKE')

# Add the 'ALIKE' directory path to sys.path
sys.path.append(thirdparty_path)

# Get the path to the 'SuperPoint' directory within 'thirdparty'
thirdparty_path = os.path.join(os.path.dirname(__file__), 'SuperPoint')

# Add the 'SuperPoint' directory path to sys.path
sys.path.append(thirdparty_path)
