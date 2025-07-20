# src/back_translation/back_translation_main.py

from translate_and_judge import main as translate_and_judge_main
from reformat import main as reformat_main

if __name__ == "__main__":
    print("Step 1: Running translate_and_judge...")
    translate_and_judge_main()
    print("Step 2: Running reformat...")
    reformat_main()