import gdelt

# Version 1 queries
gd1 = gdelt.gdelt(version=1)

# pull single day, gkg table
results= gd1.Search(['2024 Nov 01', '2025 Nov 01'],table='events')
print(len(results))
print(results.columns)
print(results.head())
# df = results[['DATE','LOCATIONS','TONE', 'CAMEOEVENTIDS']]
# print(df.head())
# # pull events table, range, output to json format
# results = gd1.Search(['2016 Oct 31','2016 Nov 2'],coverage=True,table='events')
# print(len(results))
# print(results.columns)