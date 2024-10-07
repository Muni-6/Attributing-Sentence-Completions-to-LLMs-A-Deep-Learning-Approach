import subprocess
import re
import csv
from concurrent.futures import ThreadPoolExecutor

# List of models to use
models = [
    "llama3.2:1b",
    "qwen2.5:3b",
    "mistral-openorca",
    "phi3.5",
    "granite-code",
    "codegemma"
]

# Function to read sentence starters from a file
def read_sentence_starters(input_file):
    with open(input_file, 'r') as file:
        sentence_starters = [line.strip() for line in file.readlines() if line.strip()]
    return sentence_starters

# Function to generate completions using different LLMs
def generate_sentence_completion(sentence_starts, model, word_limit=20, output_file="output_6llms.csv"):
    for sentence in sentence_starts:
        # Explicit prompt for text completion
        prompt = f'This is text completion. Complete the second part of the sentence, you can elaborate on the second part of the sentence and use minimum of 7 words. Do not repeat the first part, just give the second part only and make the first letter of the second sentence lower case. The first part of the sentence is: "{sentence}"'
        
        # Use subprocess to call the specific model and generate the completion
        command = f"ollama run {model} \"{prompt}\""
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Get the completion
            completion = result.stdout.strip()

            # Split the completion into words and limit the number of words
            words = completion.split()
            limited_completion = " ".join(words[:word_limit])

            # Option: Truncate completion at the first sentence-ending punctuation
            sentence_ending_match = re.search(r'([.!?])', limited_completion)
            if sentence_ending_match:
                # Truncate at the first sentence-ending punctuation
                limited_completion = limited_completion[:sentence_ending_match.end()]

            # Write the result to the CSV file immediately
            with open(output_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([sentence, limited_completion, model])
        else:
            print(f"Error generating completion for {sentence} using {model}")
            print(result.stderr)
    
    print(f"Completions for {model} saved to {output_file}")

# Main function to generate completions for all models with thread management
def main():
    input_file = "truncated_sentences.txt"  # Path to your input file containing sentence starters
    word_limit = 30  # Adjust word limit if needed
    output_file = "Classified_Dataset.csv"  # Combined output file for all models
    max_threads = 8  # Set the maximum number of threads (or models) to run concurrently

    # Read the sentence starters from the input file
    sentence_starts = read_sentence_starters(input_file)

    # Write header to the CSV file before starting threads
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["xi", "xj", "model"])  # CSV header

    # Use ThreadPoolExecutor to limit the number of concurrent threads
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit the generate_sentence_completion function for each model
        futures = [executor.submit(generate_sentence_completion, sentence_starts, model, word_limit, output_file) for model in models]

        # Wait for all futures to complete
        for future in futures:
            future.result()  # This will block until the thread completes

# Call the main function
if __name__ == "__main__":
    main()
