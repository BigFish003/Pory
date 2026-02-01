import json

# Open the file in read mode to load data
with open(r"C:\Users\samth\Downloads\sample_map.json", 'r') as f:
    maps = json.load(f)

# Modify the tiles
for i in range(len(maps["maps"][0]["tiles"])):
    if i in (95, 26, 59, 90, 30):
        maps["maps"][0]["tiles"][i]["improvement"] = "City"
    elif i in (0, 10, 120, 109):
        maps["maps"][0]["tiles"][i]["improvement"] = "Lighthouse"
    else:
        maps["maps"][0]["tiles"][i]["improvement"] = "None"
    if i == 95:
        maps["maps"][0]["tiles"][i]["improvement owner"] = "1"
    if i == 26:
        maps["maps"][0]["tiles"][i]["improvement owner"] = "2"
# Open the file in write mode to save changes
with open(r"C:\Users\samth\Downloads\sample_map.json", 'w') as f:
    json.dump(maps, f, indent=4)
