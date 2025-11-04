ML Hackathon: Hybrid HMM-RL Hangman Solver
This project contains a Python-based intelligent agent designed to strategically solve the game of Hangman. It was developed for the "ML Hackathon" and uses a sophisticated hybrid approach, combining a powerful statistical model with a Reinforcement Learning (Q-learning) agent to maximize its win rate.

This README provides instructions for setting up the environment, loading the data, and running the notebook.

🚀 Setup and Installation
This project is designed to run in a Google Colab environment, as indicated by the file paths in the notebook (e.g., /content/).

1. Environment
Python 3.x

Google Colab (Recommended) or a local Jupyter environment.

2. Dependencies
The project requires one main external library, numpy, for numerical operations. The other modules (pickle, collections, random, re, typing) are part of the Python standard library.

You can install numpy using pip:

Bash

pip install numpy
3. Data Files
The agent requires two text files to run: a large training corpus and a smaller test set. You must upload these files to your environment's session storage.


corpus.txt: This is the main training wordlist. The agent uses this file to build all its statistical models (letter frequencies, n-grams, etc.).


test.txt: This is the evaluation wordlist. The agent's final performance is scored against the words in this file.

In Google Colab:

Click the "Files" icon in the left-hand sidebar.

Click the "Upload to session storage" icon.

Select and upload corpus.txt and test.txt.

Ensure the files appear in the root (/content/) directory.

🏃 How to Run
Complete Setup: Follow all steps in the "Setup and Installation" section above.


Run the Main Cell: Execute the first (and largest) code cell in the ML_Hackathon_final.ipynb notebook. This cell will:

Import all libraries and define the three main classes.

Step 1: Train the AdvancedHMMHangman model on corpus.txt.

Step 2: Load the training and test word lists.

Step 3: Train the ImprovedRLAgent for 15,000 episodes.

Step 4: Evaluate the trained agent against all 2,000 words in test.txt and print the final score.

Step 5: Attempt to save the model (see "Troubleshooting" section below).

Run the Demo Cells:

Execute the second cell to define the play_game_demo helper function.

Execute the third cell to watch the agent play a game with a random word from the test set.

🧠 Model Architecture
The agent's intelligence comes from a hybrid system that combines two key components:

1. AdvancedHMMHangman (Statistical Model)
This class is the statistical backbone of the agent. It is trained on the corpus.txt file and calculates multiple probability distributions:

Word List Filtering (Primary Strategy): Given a masked word (e.g., _ _ a _ _) and guessed letters, it builds a regular expression to find all possible matching words in its dictionary. It then calculates the frequency of the next best letter from this filtered subset.

Statistical Fallback (Secondary Strategy): If no words match, it falls back to a weighted average of:

Positional Probabilities: The probability of a letter appearing at a specific index (e.g., 'e' is common at the end of a word).

N-gram Frequencies: Bigram and trigram models to predict a letter based on its known neighbors (e.g., q is almost always followed by u).

Global Frequencies: The overall frequency of letters in the entire corpus.

2. ImprovedRLAgent (Q-Learning Agent)
This is the "brain" or decision-making component. It's a Q-learning agent that learns the value of guessing a specific letter in a specific game state (where state = masked_word:lives_left).

Crucially, this agent does not learn in a vacuum. Its choose_action method combines its learned Q-values with the real-time predictions from the AdvancedHMMHangman model. This allows the agent to learn game-specific strategies (like when to guess a vowel vs. a consonant) while still relying on the powerful statistical model for letter predictions.
