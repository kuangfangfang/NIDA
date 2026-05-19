#!/usr/bin/env python3
"""
NetGuard AI — Live Network Traffic Capture & Detection
Captures live network packets using Scapy and sends them to the
NetGuard Flask backend for intrusion detection analysis.

Usage:
    sudo python live_capture.py                  # 30-second capture (default)
    sudo python live_capture.py --duration 60    # 60-second capture
    sudo python live_capture.py --duration 10 --backend http://192.168.1.100:5000

Requires: scapy, requests  (pip install scapy requests)
Requires: Administrator / root privileges for raw packet capture.
"""

import argparse
import os
import sys
import time
import json
import signal
import threading

try:
    from scapy.all import sniff, wrpcap, IP
except ImportError:
    print("✗ Error: scapy is not installed.")
    print("  Install it with:  pip install scapy")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("✗ Error: requests is not installed.")
    print("  Install it with:  pip install requests")
    sys.exit(1)


# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_DURATION = 30          # seconds
DEFAULT_BACKEND  = "http://localhost:5000"
DETECT_ENDPOINT  = "/detect"
TEMP_PCAP_FILE   = "temp_capture.pcap"

# ANSI colour helpers (works on most modern terminals)
class Colors:
    HEADER  = "\033[95m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"


# ── Banner ─────────────────────────────────────────────────────────────────────
def print_banner():
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
    ╔══════════════════════════════════════════════════╗
    ║        NetGuard AI — Live Capture Module         ║
    ║     Real-time Network Traffic Analysis Tool      ║
    ╚══════════════════════════════════════════════════╝
{Colors.RESET}"""
    print(banner)


# ── Packet capture ─────────────────────────────────────────────────────────────
class LiveCapture:
    """Manages live packet capture with real-time counter display."""

    def __init__(self, duration: int, pcap_path: str):
        self.duration = duration
        self.pcap_path = pcap_path
        self.packets = []
        self.stop_event = threading.Event()
        self._packet_count = 0
        self._lock = threading.Lock()

    # Scapy packet callback — filters for IP packets only
    def _packet_handler(self, pkt):
        if pkt.haslayer(IP):
            with self._lock:
                self.packets.append(pkt)
                self._packet_count += 1

    # Counter display thread — updates the terminal every second
    def _display_counter(self):
        start = time.time()
        while not self.stop_event.is_set():
            elapsed = time.time() - start
            remaining = max(0, self.duration - elapsed)
            with self._lock:
                count = self._packet_count
            # Overwrite the same line
            sys.stdout.write(
                f"\r{Colors.BLUE}⏳ Capturing... "
                f"{Colors.BOLD}{count}{Colors.RESET}{Colors.BLUE} packets captured  |  "
                f"{remaining:.0f}s remaining{Colors.RESET}   "
            )
            sys.stdout.flush()
            if remaining <= 0:
                break
            time.sleep(1)

    def start(self):
        """Run the capture for the configured duration. Returns packet count."""
        print(f"{Colors.GREEN}▶ Starting capture for {Colors.BOLD}{self.duration}s{Colors.RESET}"
              f"{Colors.GREEN} — press Ctrl+C to stop early.{Colors.RESET}")
        print(f"{Colors.DIM}  Filter: IP packets (TCP / UDP / ICMP / …){Colors.RESET}\n")

        # Start the display thread
        display_thread = threading.Thread(target=self._display_counter, daemon=True)
        display_thread.start()

        try:
            sniff(
                prn=self._packet_handler,
                timeout=self.duration,
                store=False,
                filter="ip",             # BPF filter — capture only IP packets
                stop_filter=lambda _: self.stop_event.is_set(),
            )
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_event.set()
            display_thread.join(timeout=2)

        # Final count
        count = len(self.packets)
        print(f"\n\n{Colors.GREEN}✓ Capture finished — {Colors.BOLD}{count}{Colors.RESET}"
              f"{Colors.GREEN} IP packets collected.{Colors.RESET}\n")
        return count

    def save(self):
        """Write captured packets to a PCAP file."""
        if not self.packets:
            return False
        wrpcap(self.pcap_path, self.packets)
        size_kb = os.path.getsize(self.pcap_path) / 1024
        print(f"{Colors.CYAN}💾 Saved to {Colors.BOLD}{self.pcap_path}{Colors.RESET}"
              f"{Colors.CYAN} ({size_kb:.1f} KB){Colors.RESET}")
        return True


# ── Detection ──────────────────────────────────────────────────────────────────
def send_to_backend(pcap_path: str, backend_url: str) -> dict | None:
    """POST the PCAP file to the Flask /detect endpoint."""
    url = backend_url.rstrip("/") + DETECT_ENDPOINT
    print(f"\n{Colors.BLUE}📡 Sending PCAP to backend: {Colors.BOLD}{url}{Colors.RESET}")

    try:
        with open(pcap_path, "rb") as f:
            files = {"file": ("temp_capture.pcap", f, "application/octet-stream")}
            response = requests.post(url, files=files, timeout=120)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"\n{Colors.RED}✗ Backend returned HTTP {response.status_code}{Colors.RESET}")
            try:
                err = response.json()
                print(f"  Error: {err.get('error', json.dumps(err, indent=2))}")
            except Exception:
                print(f"  Response body: {response.text[:500]}")
            return None

    except requests.exceptions.ConnectionError:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ Could not connect to backend at {backend_url}{Colors.RESET}")
        print(f"  {Colors.YELLOW}Make sure the Flask server is running:{Colors.RESET}")
        print(f"  {Colors.DIM}  python app.py{Colors.RESET}")
        return None
    except requests.exceptions.Timeout:
        print(f"\n{Colors.RED}✗ Request timed out. The backend may be overloaded.{Colors.RESET}")
        return None
    except Exception as e:
        print(f"\n{Colors.RED}✗ Unexpected error: {e}{Colors.RESET}")
        return None


def print_result(result: dict):
    """Pretty-print the detection result from the backend."""
    print(f"\n{'═' * 54}")
    print(f"{Colors.BOLD}  NetGuard AI — Detection Result{Colors.RESET}")
    print(f"{'═' * 54}\n")

    is_attack = result.get("is_attack", False)
    attack_info = result.get("attack_info", {})
    label = attack_info.get("label", "Unknown")
    summary = attack_info.get("summary", "")

    if is_attack:
        attack_type = result.get("attack_type", "Unknown")
        s1 = result.get("stage1_confidence", 0)
        s2 = result.get("stage2_confidence", 0)
        flagged = result.get("flagged_flows", 0)
        total = result.get("total_flows", 0)

        print(f"  {Colors.RED}{Colors.BOLD}⚠  THREAT DETECTED: {label}{Colors.RESET}")
        print(f"  {Colors.DIM}{'─' * 48}{Colors.RESET}")
        print(f"  {Colors.YELLOW}Stage 1 confidence : {s1:.2%}{Colors.RESET}")
        print(f"  {Colors.YELLOW}Stage 2 confidence : {s2:.2%}{Colors.RESET}")
        print(f"  {Colors.YELLOW}Flagged flows      : {flagged} / {total}{Colors.RESET}")
        print()
        print(f"  {Colors.BOLD}Summary:{Colors.RESET}")
        # Word-wrap the summary at ~60 chars
        words = summary.split()
        line = "  "
        for w in words:
            if len(line) + len(w) + 1 > 62:
                print(line)
                line = "  " + w
            else:
                line += " " + w if line.strip() else "  " + w
        if line.strip():
            print(line)

        suggestions = attack_info.get("suggestions", [])
        if suggestions:
            print(f"\n  {Colors.BOLD}Recommended Actions:{Colors.RESET}")
            for i, s in enumerate(suggestions, 1):
                print(f"  {Colors.CYAN}{i}. {s}{Colors.RESET}")
    else:
        s1 = result.get("stage1_confidence", 0)
        total = result.get("total_flows", 0)
        print(f"  {Colors.GREEN}{Colors.BOLD}✓  {label}{Colors.RESET}")
        print(f"  {Colors.DIM}{'─' * 48}{Colors.RESET}")
        print(f"  {Colors.GREEN}Confidence : {s1:.2%}{Colors.RESET}")
        print(f"  {Colors.GREEN}Flows      : {total}{Colors.RESET}")
        print(f"\n  {summary}")

    if result.get("demo_mode"):
        print(f"\n  {Colors.YELLOW}⚡ Demo mode — models not loaded, results are simulated.{Colors.RESET}")

    # Class distribution (if available)
    dist = result.get("class_distribution")
    if dist:
        print(f"\n  {Colors.BOLD}Class Distribution:{Colors.RESET}")
        for cls, cnt in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"    {cls:<20} {cnt}")

    print(f"\n{'═' * 54}\n")


# ── Cleanup ────────────────────────────────────────────────────────────────────
def cleanup(pcap_path: str):
    """Remove the temporary PCAP file."""
    try:
        if os.path.exists(pcap_path):
            os.remove(pcap_path)
            print(f"{Colors.DIM}🗑  Temporary file removed: {pcap_path}{Colors.RESET}")
    except OSError as e:
        print(f"{Colors.YELLOW}⚠ Could not delete temp file: {e}{Colors.RESET}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="NetGuard AI — Live traffic capture and intrusion detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python live_capture.py                          # 30s capture, send to localhost:5000
  python live_capture.py --duration 60            # 60s capture
  python live_capture.py --backend http://10.0.0.5:5000
  python live_capture.py --duration 15 --no-cleanup   # keep the pcap file
        """
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=DEFAULT_DURATION,
        help=f"Capture duration in seconds (default: {DEFAULT_DURATION})"
    )
    parser.add_argument(
        "--backend", "-b",
        type=str,
        default=DEFAULT_BACKEND,
        help=f"Backend URL (default: {DEFAULT_BACKEND})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=TEMP_PCAP_FILE,
        help=f"Temporary PCAP filename (default: {TEMP_PCAP_FILE})"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep the temporary PCAP file after detection"
    )
    args = parser.parse_args()

    print_banner()

    # ── Pre-flight checks ──────────────────────────────────────────────────
    print(f"{Colors.DIM}  Backend  : {args.backend}{Colors.RESET}")
    print(f"{Colors.DIM}  Duration : {args.duration}s{Colors.RESET}")
    print(f"{Colors.DIM}  PCAP file: {args.output}{Colors.RESET}")
    print()

    # Check backend availability first
    try:
        health_url = args.backend.rstrip("/") + "/health"
        resp = requests.get(health_url, timeout=5)
        if resp.status_code == 200:
            info = resp.json()
            mode = "Production" if info.get("models_loaded") else "Demo (models not loaded)"
            print(f"{Colors.GREEN}✓ Backend is online — mode: {mode}{Colors.RESET}\n")
        else:
            print(f"{Colors.YELLOW}⚠ Backend responded with HTTP {resp.status_code}. "
                  f"Proceeding anyway…{Colors.RESET}\n")
    except requests.exceptions.ConnectionError:
        print(f"{Colors.RED}{Colors.BOLD}✗ Cannot reach backend at {args.backend}{Colors.RESET}")
        print(f"  {Colors.YELLOW}Start the Flask server first:{Colors.RESET}")
        print(f"  {Colors.DIM}  cd {os.path.dirname(os.path.abspath(__file__))}{Colors.RESET}")
        print(f"  {Colors.DIM}  python app.py{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"{Colors.YELLOW}⚠ Could not check backend health: {e}. Proceeding…{Colors.RESET}\n")

    # ── Phase 1: Capture ───────────────────────────────────────────────────
    capture = LiveCapture(duration=args.duration, pcap_path=args.output)

    # Handle Ctrl+C gracefully during capture
    original_sigint = signal.getsignal(signal.SIGINT)

    def sigint_handler(sig, frame):
        print(f"\n\n{Colors.YELLOW}⚠ Ctrl+C detected — stopping capture…{Colors.RESET}")
        capture.stop_event.set()

    signal.signal(signal.SIGINT, sigint_handler)

    packet_count = capture.start()

    # Restore original handler
    signal.signal(signal.SIGINT, original_sigint)

    if packet_count == 0:
        print(f"{Colors.RED}✗ No packets captured. Possible causes:{Colors.RESET}")
        print(f"  {Colors.YELLOW}• Not running with administrator/root privileges{Colors.RESET}")
        print(f"  {Colors.YELLOW}• No active network interfaces{Colors.RESET}")
        print(f"  {Colors.YELLOW}• Capture duration too short{Colors.RESET}")
        sys.exit(1)

    # Save to PCAP
    if not capture.save():
        print(f"{Colors.RED}✗ Failed to save PCAP file.{Colors.RESET}")
        sys.exit(1)

    # ── Phase 2: Detection ─────────────────────────────────────────────────
    try:
        result = send_to_backend(args.output, args.backend)
        if result:
            print_result(result)
        else:
            print(f"\n{Colors.RED}✗ Detection failed — see errors above.{Colors.RESET}")
    finally:
        # ── Phase 3: Cleanup ───────────────────────────────────────────────
        if not args.no_cleanup:
            cleanup(args.output)
        else:
            print(f"{Colors.DIM}ℹ  PCAP file kept: {args.output}{Colors.RESET}")

    print(f"{Colors.GREEN}{Colors.BOLD}Done.{Colors.RESET}")


if __name__ == "__main__":
    main()
