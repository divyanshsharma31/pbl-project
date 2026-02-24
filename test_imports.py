#!/usr/bin/env python
"""Quick test to verify all imports work"""

try:
    import streamlit as st
    print("✓ streamlit imported")
    
    import pandas as pd
    print("✓ pandas imported")
    
    import numpy as np
    print("✓ numpy imported")
    
    import matplotlib.pyplot as plt
    print("✓ matplotlib imported")
    
    import seaborn as sns
    print("✓ seaborn imported")
    
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    print("✓ sklearn imported")
    
    # Try to syntax check app.py
    import ast
    with open('app.py', 'r', encoding='utf-8') as f:
        code = f.read()
        ast.parse(code)
    print("✓ app.py syntax is valid")
    
    print("\n✅ All imports successful!")
    
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
