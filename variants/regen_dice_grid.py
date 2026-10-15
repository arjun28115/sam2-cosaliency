"""Dice regen with GRID prompts (an existing pipeline mode) + small-object dials.
Diagnosis: die-level SAM2 masks exist (iou .97) but centroid prompts never land
on dice (SNIC merges them into felt at 64x64 embed resolution), and
MIN_AREA_FRAC=0.002 (= 2x die area) erases any die club in cleanup."""
import sys
sys.path.insert(0, "/Users/arjunsingla/sam2")
import pipeline_final as pf

pf.PROMPT_MODE = "grid"     # dense uniform prompts reach dice-sized objects
pf.GRID = 32                # 1024 prompts
pf.MIN_AREA_FRAC = 0.0005   # keep die-sized clubs (0.05% of image)
pf.BASE_N_SP = 600          # finer pooling superpixels
pf.SP_CAP = 900

sys.argv = ["pipeline_final.py",
            "--image_dir", "/Users/arjunsingla/Desktop/Images/dice",
            "--output_dir", "/tmp/dice_grid"]
pf.main()
