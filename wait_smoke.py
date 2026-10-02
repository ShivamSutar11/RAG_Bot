import time
import sys
log_file = "/Users/shivamsutar/.gemini/antigravity/brain/ae54c97e-f5fb-45b0-b81a-5c6201505fe4/.system_generated/tasks/task-1232.log"
while True:
    try:
        with open(log_file, "r") as f:
            content = f.read()
            if "Saved to evals/ragas_report_smoke.md" in content:
                print("Completed successfully!")
                sys.exit(0)
            if "Traceback" in content or "Exception raised in Job" in content:
                print("Error detected!")
                sys.exit(1)
    except FileNotFoundError:
        pass
    time.sleep(5)
