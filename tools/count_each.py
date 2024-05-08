import json

# Load the JSON file
with open('./data/coco/City2Foggy_source/annotations/instances_train2017.json', 'r') as f:
    data = json.load(f)

# Dictionary to store the count of instances for each class
class_counts = {}

# Get category names
category_names = {}
for category in data['categories']:
    category_names[category['id']] = category['name']

# Iterate through annotations
for annotation in data['annotations']:
    category_id = annotation['category_id']
    # The category_id corresponds to the class
    if category_id not in class_counts:
        class_counts[category_id] = 1
    else:
        class_counts[category_id] += 1

# Print the counts for each class along with their names
for category_id, count in class_counts.items():
    class_name = category_names.get(category_id, 'Unknown')  # Get class name or 'Unknown' if not found
    print(f"Class {category_id} ({class_name}): {count} instances")

