import os

# Path to the folder where the score files are located
score_folder = "scores"

# Iterate over the files in the score folder
for file_name in os.listdir(score_folder):
    if file_name.endswith(".txt"):  # Check if it's a text file
        file_path = os.path.join(score_folder, file_name)
        with open(file_path, "r") as file:
            lines = file.readlines()  # Read all lines from the file
            # Extract the score from each line and convert to float
            scores = [float(line.split(":")[-1].strip()) for line in lines if line.startswith("Round")]
            # Calculate the average score
            average_score = sum(scores) / len(scores)
            # Round the average score to 2 decimal places
            average_score = round(average_score, 2)
            # Print the average score of the file
            print(f"{file_name}: {average_score}")
