import lightkurve as lk
from lightkurve import search_lightcurve
import matplotlib.pyplot as plt
import pandas as pd


# use TIC or target name
target = "TIC 39903405"   # HAT-P-2
print("Searching for pipeline light curves (SPOC/QLP)...")
sr = search_lightcurve(target, mission="TESS")   # returns a SearchResult
print(sr)

lc_collection = lk.search_lightcurve("TIC 39903405", mission="TESS", author="SPOC", exptime=120).download_all()
print(lc_collection)

# Stitch into one combined light curve
lc = lc_collection.PDCSAP_FLUX.stitch().remove_nans()
# Convert to pandas DataFrame
df = lc.to_pandas()
print(df)

lc.plot()
plt.show()

# Save as CSV
df.to_csv("hatp2_tess_lightcurve.csv")