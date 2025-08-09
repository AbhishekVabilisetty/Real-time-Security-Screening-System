# alerts.py

import time

try:
    import serial
except Exception:
    serial = None

try:
    import winsound
    def beep(freq, dur): winsound.Beep(freq, dur)
except Exception:
    def beep(freq, dur): pass  # no-op on non-Windows

def init_arduino(port, baud):
    if serial is None:
        print("[INFO] pyserial not installed; Arduino disabled.")
        return None
    try:
        arduino = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        print("[INFO] Arduino connected on", port)
        return arduino
    except Exception as e:
        print("[WARN] Arduino connect failed:", e)
        return None

def alert(arduino, beep_freq, beep_dur):
    try:
        beep(beep_freq, beep_dur)
    except Exception:
        pass
    if arduino:
        try:
            arduino.write(b"ALERT\n")
            print("[INFO] ALERT sent to Arduino")
        except Exception as e:
            print("[WARN] Arduino send failed:", e)
