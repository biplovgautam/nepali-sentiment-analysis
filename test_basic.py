#!/usr/bin/env python3
"""Test just basic imports without execution."""

try:
    print("Testing config...")
    import config
    print("✅ config OK")
except Exception as e:
    print(f"❌ config failed: {e}")

try:
    print("Testing loader...")
    from utils.loader import load_dataset
    print("✅ loader OK")
except Exception as e:
    print(f"❌ loader failed: {e}")

print("Basic test completed!")
