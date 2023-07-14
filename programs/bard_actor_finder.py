from bardapi import Bard
import pandas as pd
import gpt4all

bard_token_path = '../bardapi_token.txt'

with open(bard_token_path, 'r') as file:
    token = file.read()
    bard = Bard(token=token)

imdb_movies = pd.read_csv('../data/imdb_movies.csv', index_col="names")

answers = []  # Create an empty list to store the answers

for index, row in imdb_movies.iterrows():
    crew = row['crew']
    name = row['orig_title']
    prompt = "Given the crew of a movie, identify only the most famous actor, meaning the actor " \
             "that is the most identifiable and recognizable to any given person in the public. Output ONLY the " \
             "actor's name and nothing else." \
             "Example:" \
             "Movie Name: The Super Mario Bros. Movie" \
             "Crew: Chris Pratt, Mario (voice), Anya Taylor-Joy, Princess Peach (voice), Charlie Day, Luigi (voice), " \
             "Jack Black, Bowser (voice), Keegan-Michael Key, Toad (voice), Seth Rogen, Donkey Kong (voice), " \
             "Fred Armisen, Cranky Kong (voice), Kevin Michael Richardson, Kamek (voice), Sebastian Maniscalco, " \
             "Spike (voice)" \
             "Output: The most famous actor is Chis Pratt" \
             "Here is your input:" \
             f"Movie Name: {name}" \
             f"Crew: {crew}"

    # Send an API request to Bard and get the response
    response = bard.get_answer(prompt)

    # Get the model's reply
    reply = response['content']

    answers.append(reply)

    print(answers)
