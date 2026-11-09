from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parent
def run(*args):
    print("\n>>>", " ".join(args), flush=True)
    subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
if __name__=="__main__":
    run("generate_data.py")
    run("src/part_a_data.py")
    run("src/part_b_ml.py")
    run("src/part_c_triage.py")
    run("src/part_d_agent.py")
    print("\nALL PIPELINE STEPS COMPLETED. Review outputs/ and README.md.")
