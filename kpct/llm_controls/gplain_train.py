#!/usr/bin/env python3
"""gplain_train.py -- GPLAIN_1 (amendment LC1): the frozen gk_train.py with the plain-evidence renderer patched in
(gk_plain.patch(): gk_common.fit_g_prompt -> fit_plain_prompt, g_ranked -> alphabetical tail, prompt hashes).
All arguments are passed through unchanged to gk_train.main(); use --arm G."""
import os, sys
sys.path.insert(0, os.environ.get("GK_CODE", "/kpct/guided/code"))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import gk_common as GC
import gk_plain
gk_plain.patch()
import gk_train

if __name__ == "__main__":
    assert GC.fit_g_prompt is gk_plain.fit_plain_prompt
    gk_train.main()
