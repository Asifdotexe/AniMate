"""
Module that contains the test cases for the conftest module.
"""

import os
import sys

# Add the project root to sys.path so we can import src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add webscraping directory to sys.path
webscraping_dir = os.path.join(project_root, "code", "webscraping")
if webscraping_dir not in sys.path:
    sys.path.append(webscraping_dir)
