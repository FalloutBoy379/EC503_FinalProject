import csv
import matplotlib.pyplot as plt
import numpy as np

# Specify the path to your CSV file
csv_file_path = './Life Expectancy Data.csv'

# Open the CSV file
with open(csv_file_path, 'r') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)
    
    # Initialize an empty list to store the labels
    features = []

    # Read the first row of the CSV file
    header = next(csv_reader)

    # Store the labels in the list
    features = header

    # Initialize an empty dictionary to store the countries and their corresponding labels
    country_labels = {}

    # Read each row of the CSV file
    for row in csv_reader:
        # Extract the country from the first column
        country = row[0]
        
        # Add the country to the dictionary if it's not already present
        if country not in country_labels:
            # Initialize an empty dictionary to store the labels for each year
            country_labels[country] = {}
        
        # Extract the year from the second column
        year = row[1]
        
        # Extract the labels from the remaining columns
        labels = row[2:]
        
        # Add the labels to the dictionary for the country and year
        country_labels[country][year] = labels
    
    # Print the labels for each country and year
    print("Country Labels:\n")
    for country, years in country_labels.items():
        print(country)
        for year, labels in years.items():
            print(year, labels)
        print()

        # Calculate the correlation between features for every country
        for country, years in country_labels.items():
            print(f"Correlation for {country}:")
            for year, labels in years.items():
                # Convert labels to numeric values
                numeric_labels = []
                for label in labels:
                    try:
                        numeric_labels.append(float(label))
                    except ValueError:
                        numeric_labels.append(0.0)  # Replace non-numeric labels with 0.0
                                
                # Calculate the correlation matrix
                correlation_matrix = np.corrcoef(numeric_labels)
                                
                # Print the correlation matrix
                print(f"Year {year}:")
                print(correlation_matrix)
                print()
                     

                # Visualize the correlation matrix
                plt.imshow(correlation_matrix, cmap='hot', interpolation='nearest')
                plt.colorbar()
                plt.title(f"Correlation Matrix for {country} - Year {year}")
                plt.xlabel("Features")
                plt.ylabel("Features")
                # Visualize the correlation matrix
                if correlation_matrix.shape[0] > 0 and correlation_matrix.shape[1] > 0:
                    plt.imshow(correlation_matrix, cmap='hot', interpolation='nearest')
                    plt.colorbar()
                    plt.title(f"Correlation Matrix for {country} - Year {year}")
                    plt.xlabel("Features")
                    plt.ylabel("Features")
                    plt.show()
