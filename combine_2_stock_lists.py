import os

# File paths
file1_path = 'stocks_list2.txt'
file2_path = '../csv_column_extractort/SP_TOP5-.csv'
output_file_path = 'stocks_list5.txt'

def read_file(file_path):
    """Read file and return a set of cleaned stock symbols."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return set()
    
    with open(file_path, 'r') as file:
        content = file.read()
        # Split the content by comma, strip whitespace and newline characters
        symbols = {symbol.strip() for symbol in content.split(',')}
    return symbols

def write_file(file_path, data):
    """Write data to file."""
    with open(file_path, 'w') as file:
        file.write(data)

# Read and clean symbols from both files
symbols1 = read_file(file1_path)
symbols2 = read_file(file2_path)

# Combine lists and remove duplicates
combined_symbols = symbols1.union(symbols2)

# Sort the set alphabetically
sorted_symbols = sorted(combined_symbols)

# Convert the sorted set back to a comma-separated string
result = ','.join(sorted_symbols)

# Write the result to the output file
write_file(output_file_path, result)

print(f"Combined stock symbols written to {output_file_path}")
