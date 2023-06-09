# %% [markdown]
# # Dataset Pre-Processing Notebook
# The purpose of this notebook is to preprocess each file of the datasets we collected.
# We want all the dataset as a single csv-file with stories as entries.

# %%
import os

import logging as log
import pathlib

# %% [markdown]
# ### Settings

# %%
log.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=log.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

log_enabled = True
show_notebook_results = True

# %% [markdown]
# # Experimental

# %%
CUT_OFF_LINE = 10000

def txt_files_to_lines_gen(file_path):
    """
    Generator function that yields lines from text files in a directory.

    Args:
        file_path (str):    Path to the directory containing the text files.

    Yields:
        str:                A line from a text file.
    """
    log.debug(f'execute')
    path = pathlib.Path(file_path)

    for file in path.iterdir():
        if file.is_file():
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    yield line.strip()

def lines_to_fit_sentences(sentences, length):
        """
        Generator function that combines sentences so that the combined sentence is close to a certain length.
        
        Args:
            sentences (iterator):   An iterator that yields sentences.
            length (int):           The maximum length for combined sentences.
            
        Yields:
            str:                    A combined sentence.
        """
        log.debug(f'execute')
        length = length / 1.5 # estimate of token/word ratio (in real the value is about 1.4)

        current_combined_sentence = ""

        for sentence in sentences:
            sentence = sentence.strip()  # Remove leading/trailing whitespace
            sentence_words = sentence.split()

            # Check if combining the current sentence with the previous one exceeds the word limit
            if len(current_combined_sentence.split()) + len(sentence_words) > length:
                yield current_combined_sentence
                current_combined_sentence = sentence  # Start a new combined sentence
            else:
                current_combined_sentence += " " + sentence  # Concatenate the sentences

def process_and_save_txt_files(input_path, output_path, sentence_length, cutoff_line):
    """
    Loads all text files from a directory, processes them into fitting sentences,
    and saves them into multiple text files.

    Args:
        input_path (str):      The path to the directory with the text files to process.
        output_path (str):     The path where the processed text files will be saved.
        sentence_length (int): The maximum length for the processed sentences.
        cutoff_line (int):     The number of sentences per output file.
    """
    log.debug(f'Start processing text files from {input_path} into sentences of length {sentence_length}.')

    # Create output directory if it doesn't exist
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate lines from all text files in the directory
    lines = txt_files_to_lines_gen(input_path)

    # Combine lines into sentences of fitting length
    sentences = lines_to_fit_sentences(lines, sentence_length)

    # Prepare to count sentences
    count = 0

    # Prepare to count files
    file_count = 0

    # Prepare to write to an output file
    f = open(os.path.join(output_path, f'output_{file_count}.txt'), 'w', encoding='utf-8')

    try:
        # Iterate over the sentences generator
        for sentence in sentences:
            # Write the sentence to the file
            f.write(sentence + '\n')

            # Increase the count of sentences
            count += 1

            # If we have reached the cutoff line, start a new output file
            if count >= cutoff_line:
                log.debug(f'Saved {cutoff_line} sentences into file {file_count}.txt.')

                # Close the current file
                f.close()

                # Reset the sentence count
                count = 0

                # Increase the file count
                file_count += 1

                # Open a new file for writing
                f = open(os.path.join(output_path, f'output_{file_count}.txt'), 'w')
    finally:
        # Make sure the last file gets closed
        f.close()

    log.info(f'Finished processing text files. Generated {file_count} output files.')
    
process_and_save_txt_files(input_path='datasets/corpus', 
                           output_path='datasets/corpus/processed_512', 
                           sentence_length=512, 
                           cutoff_line=10000)
# %%
#process_and_save_txt_files(input_path='datasets/bookscorpusopen/epubtxt', 
#                           output_path='datasets/bookscorpusopen/processed_512', 
#                           sentence_length=512, 
#                           cutoff_line=10000)


