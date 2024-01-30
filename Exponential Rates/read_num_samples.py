import csv 
def find_num_sample(file_path = 'dataset_exp.csv'): 
    # Open the CSV file and count the number of rows
        num_rows = 0
        with open(file_path, mode="r") as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                num_rows += 1
        return num_rows-1

print(find_num_sample())