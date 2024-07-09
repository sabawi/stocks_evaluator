import re

# Function to split the input string into items
def split_items(input_string):
    # Define a regular expression pattern to match delimiters and control characters
    pattern = r'[,\s\n\r;:]+'
    # Split the input string using the pattern
    return re.split(pattern, input_string)

# Prompt the user for input file paths
file1_path = input("Enter the path of the first input file: ")
file2_path = input("Enter the path of the second input file: ")

# Prompt the user for the output file name
output_file_name = input("Enter the name of the output file: ")

# Read and process the first file
with open(file1_path, 'r') as f:
    items1 = f.read()
    items1 = split_items(items1)

# Read and process the second file
with open(file2_path, 'r') as f:
    items2 = f.read()
    items2 = split_items(items2)

# Merge the two lists of items
items = set(items1 + items2)

# Sort the merged list of items
items = sorted(items)

# Write the merged list of items to the output file
with open(output_file_name, 'w') as f:
    f.write(','.join(items))

