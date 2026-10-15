"""Clover regen with grid prompts + small-object min-area (same pipeline flow,
existing PROMPT_MODE option) — targets img3's tiny corner clover recall."""
import sys
sys.path.insert(0, "/Users/arjunsingla/sam2")
import pipeline_final as pf

pf.PROMPT_MODE = "grid"
pf.GRID = 32
pf.MIN_AREA_FRAC = 0.0005

sys.argv = ["pipeline_final.py",
            "--image_dir", "/Users/arjunsingla/Desktop/Images/clover",
            "--output_dir", "/tmp/clover_grid"]
pf.main()
