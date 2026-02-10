import time

try:
    import serial
    import serial.tools.list_ports
except Exception:
    serial = None


def find_arduino_port():
    """Auto-detect Arduino COM port."""
    if serial is None:
        return None
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "Arduino" in p.description or "CH340" in p.description or "USB Serial" in p.description:
            print(f"[INFO] Auto-detected Arduino on {p.device}")
            return p.device
    if ports:
        print(f"[WARN] No explicit Arduino port found; using {ports[0].device}")
        return ports[0].device
    print("[WARN] No serial ports detected.")
    return None


def init_arduino(port=None, baud=9600):
    """Initialize Arduino serial connection (auto-detect if port not given)."""
    if serial is None:
        print("[INFO] pyserial not installed; Arduino disabled.")
        return None

    if port is None:
        port = find_arduino_port()
        if port is None:
            print("[WARN] Could not find Arduino automatically.")
            return None

    try:
        arduino = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # Wait for Arduino reset
        print("[INFO] Arduino connected on", port)
        return arduino
    except Exception as e:
        print("[WARN] Arduino connect failed:", e)
        return None


def alert(arduino, *args):
    """Send alert to Arduino (triggers buzzer)."""
    if arduino:
        try:
            arduino.write(b"ALERT\n")
            print("[INFO] ALERT sent to Arduino")
        except Exception as e:
            print("[WARN] Arduino send failed:", e)
