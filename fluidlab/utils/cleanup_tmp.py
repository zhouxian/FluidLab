import os
import shutil

if os.path.exists('/tmp/fluidlab'):
    shutil.rmtree('/tmp/fluidlab')