import gdelt
import pandas as pd


# pull single day, gkg table
today = pd.Timestamp.today().strftime('%Y %m %d')
gd2 = gdelt.gdelt(version=2)

# Single 15 minute interval pull, output to json format with mentions table
results = gd2.Search(['20260506'],table='events',output='json')
df = pd.DataFrame(results)
df.to_csv("scripts/events/gdelt_clean_mapped.csv", index=False, encoding="utf-8")