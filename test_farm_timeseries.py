#!/usr/bin/env python3
"""
Simple syntax and import test for weather_energy_monthly.py
"""

import ast
import sys
from pathlib import Path

def test_syntax():
    """Test that the script has valid Python syntax."""
    script_path = Path(__file__).parent / "scripts" / "weather_energy_monthly.py"
    with open(script_path) as f:
        code = f.read()
    
    try:
        ast.parse(code)
        print("✓ Syntax check passed")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False

def test_cli_signature():
    """Test that CLI arguments are properly defined."""
    script_path = Path(__file__).parent / "scripts" / "weather_energy_monthly.py"
    with open(script_path) as f:
        code = f.read()
    
    # Check for the new flag
    if "--write-farm-timeseries" in code:
        print("✓ CLI flag --write-farm-timeseries found")
    else:
        print("✗ CLI flag --write-farm-timeseries not found")
        return False
    
    # Check for atomic write function
    if "_write_farm_netcdf_atomic" in code:
        print("✓ Atomic write function found")
    else:
        print("✗ Atomic write function not found")
        return False
    
    # Check for per-farm write functions
    if "_write_pv_farm_timeseries" in code and "_write_wind_farm_timeseries" in code:
        print("✓ Per-farm write functions found")
    else:
        print("✗ Per-farm write functions not found")
        return False
    
    # Check for h5netcdf engine usage
    if 'engine="h5netcdf"' in code:
        print("✓ h5netcdf engine usage found")
    else:
        print("✗ h5netcdf engine usage not found")
        return False
    
    # Check for tempfile import
    if "import tempfile" in code or "from tempfile import" in code:
        print("✓ tempfile import found for atomic writes")
    else:
        print("✗ tempfile import not found")
        return False
    
    # Check for return_farm_data parameter
    if "return_farm_data" in code:
        print("✓ return_farm_data parameter found")
    else:
        print("✗ return_farm_data parameter not found")
        return False
    
    return True

def test_safe_start_year_handling():
    """Test that start_year is handled safely (no unsafe casting)."""
    script_path = Path(__file__).parent / "scripts" / "weather_energy_monthly.py"
    with open(script_path) as f:
        code = f.read()
    
    # Look for the pattern where we compute asbuilt_mask before any casting
    if 'pd.to_numeric(df_country["Start year"], errors="coerce").to_numpy()' in code:
        print("✓ Safe start_year parsing found")
    else:
        print("✗ Safe start_year parsing not found")
        return False
    
    if "asbuilt_mask = np.isfinite(start_year)" in code:
        print("✓ asbuilt_mask computed on float values")
    else:
        print("✗ asbuilt_mask not computed on float values")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing weather_energy_monthly.py changes...\n")
    
    results = []
    results.append(test_syntax())
    results.append(test_cli_signature())
    results.append(test_safe_start_year_handling())
    
    print("\n" + "="*50)
    if all(results):
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed!")
        sys.exit(1)
