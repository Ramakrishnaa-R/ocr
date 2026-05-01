#!/usr/bin/env python
"""Detect available cameras on the system."""

import cv2

print("Scanning for available camera devices...\n")

found = False
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✓ Camera found at index {i}")
        print(f"  Width: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}")
        print(f"  Height: {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        cap.release()
        found = True
    else:
        cap.release()

if not found:
    print("❌ No USB cameras detected at indices 0-9")
    print("\nTroubleshooting:")
    print("1. Check if your webcam is connected to the PC")
    print("2. Check Device Manager (Windows) → Imaging Devices for your camera")
    print("3. Try unplugging and replugging the camera")
    print("4. Check if another app is using the camera")
else:
    print("\n✓ Use the camera index shown above in test.py if needed")
