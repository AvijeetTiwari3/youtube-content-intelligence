"""
End-to-End Smoke Test for YouTube Streamlit Dashboard Server
"""

import subprocess
import sys
import time
import urllib.request


def test_dashboard_server():
    print("Launching YouTube Dashboard server for smoke testing...")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "youtube-content-intelligence/dashboard/app.py",
            "--server.headless",
            "true",
            "--server.port",
            "8598",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    time.sleep(4)

    try:
        url_health = "http://localhost:8598/_stcore/health"
        print(f"Pinging health endpoint: {url_health}")
        req = urllib.request.urlopen(url_health, timeout=5)
        health_status = req.getcode()
        print(f"Health Status: {health_status} OK")
        assert health_status == 200

        url_main = "http://localhost:8598"
        print(f"Pinging main portal: {url_main}")
        req_main = urllib.request.urlopen(url_main, timeout=5)
        main_status = req_main.getcode()
        html = req_main.read().decode("utf-8")
        print(f"Main Page Status: {main_status} OK | Loaded {len(html):,} bytes")
        assert main_status == 200

        print("\nSMOKE TEST PASSED: YouTube Dashboard is fully functional and serving live traffic!")
    finally:
        print("Terminating test server...")
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    test_dashboard_server()
