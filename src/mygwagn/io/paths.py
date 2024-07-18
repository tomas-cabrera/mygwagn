import os
import os.path as pa

import mygwagn

# Path to module
mygwagn_dir = mygwagn.__file__

# Path to data
data_dir = pa.join(pa.dirname(mygwagn_dir), "data")

# Path to cache
cache_dir = pa.join(data_dir, ".cache")
