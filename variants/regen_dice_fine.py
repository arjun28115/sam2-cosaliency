"""Regenerate the dice set at finer granularity — a SCALE dial, not category
knowledge: more superpixels + smaller min-region so tiny objects survive as
their own clubs. Tests the diagnosis that dice failed on candidate recall."""
import sys
sys.path.insert(0, "/Users/arjunsingla/sam2")
import pipeline_final as pf

pf.BASE_N_SP = 600          # 250 -> 600 superpixels (finer)
pf.SP_CAP = 900
pf.MIN_AREA_FRAC = 0.0005   # keep regions >= 0.05% of image (dice-sized)
pf.SNIC_TARGET = 256        # finer latent-SNIC grid

sys.argv = ["pipeline_final.py",
            "--image_dir", "/Users/arjunsingla/Desktop/Images/dice",
            "--output_dir", "/tmp/dice_fine"]
pf.main()
