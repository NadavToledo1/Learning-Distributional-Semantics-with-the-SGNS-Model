from ex2 import SkipGram, normalize_text
import time
import nltk
nltk.download('punkt_tab')

# print("-------------------------- drSeuss tests --------------------------")
#
# sentences = normalize_text("drSeuss.txt")
#
# # Initialize the model
# model = SkipGram(sentences, d=50, neg_samples=5, context=2, word_count_threshold=2)
#
# # Train the model
# print("Training embeddings...")
# start_time = time.time()
# model.learn_embeddings(epochs=50)
# model.V = model.combine_vectors(model.T, model.C, combo=0)
# training_time = time.time() - start_time
# print("training time")
# print("--- %s seconds ---" % training_time)
#
# # Test similarity
# word1 = "cat"
# word2 = "hat"
# sim = model.compute_similarity(word1, word2)
# similarity_time = time.time() - training_time
# print("similarity time")
# print("--- %s seconds ---" % similarity_time)
# print(f"\nSimilarity between '{word1}' and '{word2}': {sim}")
#
# # Test closest words
# print(f"\nWords closest to '{word1}':")
# for word in model.get_closest_words(word1, 5):
#     print(word)
# closest_time = time.time()-similarity_time
# print("closest time")
# print("--- %s seconds ---" % closest_time)
#
# # Test combine_vectors
# print("\nCombining embeddings with method 0...")
# V = model.combine_vectors(model.T, model.C, combo=0)
# combine_time = time.time() - closest_time
# print("combine time")
#
# print("--- %s seconds ---" % combine_time)
#
# # Test analogy: "man is to king as woman is to ?"
# w1, w2, w3 = "man", "king", "woman"
# analogy = model.find_analogy(w1, w2, w3)
# print(f"\nAnalogy: '{w1}' is to '{w2}' as '{w3}' is to '{analogy}'")
#
# analogy_time = time.time() - combine_time
# print("analogy time")
#
# print("--- %s seconds ---" % analogy_time)
# # Test analogy evaluation
# print("\nTesting analogy with ground truth...")
# result = model.test_analogy("man", "king", "woman", "queen", n=1)
# print(f"Was the analogy correct? {'Yes' if result else 'No'}")
#
# print("man" in model.word_to_index)
# print("king" in model.word_to_index)
# print("woman" in model.word_to_index)
#
# # Analogy: 'cat' is to 'hat' as 'fish' is to ?
# print("Analogy: 'cat' is to 'hat' as 'fish' is to", model.find_analogy("cat", "hat", "fish"))
#
# # Analogy: 'thing' is to '1' as 'thing' is to ?
# print("Analogy: 'thing' is to '1' as 'thing' is to", model.find_analogy("thing", "1", "thing"))
#
# # Analogy: 'in' is to 'out' as 'here' is to ?
# print("Analogy: 'in' is to 'out' as 'here' is to", model.find_analogy("in", "out", "here"))
#
# # Analogy: 'day' is to 'play' as 'night' is to ?
# print("Analogy: 'day' is to 'play' as 'night' is to", model.find_analogy("day", "play", "night"))
#
# # Analogy: 'rain' is to 'wet' as 'sun' is to ?
# print("Analogy: 'rain' is to 'wet' as 'sun' is to", model.find_analogy("rain", "wet", "sun"))
#
# print("Similarity between 'fish' and 'dish':", model.compute_similarity("fish", "dish"))
# print("Similarity between 'cat' and 'thing':", model.compute_similarity("cat", "thing"))
# print("Similarity between 'fun' and 'run':", model.compute_similarity("fun", "run"))
# print("Similarity between 'rain' and 'sun':", model.compute_similarity("rain", "sun"))
#
# print("Words closest to 'fish':", model.get_closest_words("fish"))
# print("Words closest to 'fun':", model.get_closest_words("fun"))
# print("Words closest to 'play':", model.get_closest_words("play"))
#
# print("Vocab sample:", list(model.word_to_index.keys())[:50])

print("-------------------------- harry potter 1 --------------------------")
# Step 1: Normalize Text
start = time.time()
sentences = normalize_text("harryPotter1.txt")
print("Normalization time:", time.time() - start, "seconds")

# Step 2: Train model
model = SkipGram(sentences, d=50, neg_samples=5, context=2, word_count_threshold=2)
start = time.time()
model.learn_embeddings(epochs=50)
print("Training time:", time.time() - start, "seconds")

# Step 3: Analogy Tests
print("\n--- Analogy Tests ---")
analogies = [
    ("harry", "hogwarts", "ron"),
    ("dumbledore", "wizard", "mcgonagall"),
    ("harry", "quidditch", "ron"),
    ("muggle", "magic", "hermione"),
    ("draco", "slytherin", "harry"),
    ("ron", "brother", "hermione"),
    ("owl", "letter", "train"),
    ("stone", "nicholas", "potter"),
]

for a, b, c in analogies:
    result = model.find_analogy(a, b, c)
    print(f"'{a}' is to '{b}' as '{c}' is to '{result}'")

# Step 4: Similarity Tests
print("\n--- Similarity Tests ---")
pairs = [
    ("harry", "ron"),
    ("dumbledore", "mcgonagall"),
    ("hogwarts", "school"),
    ("voldemort", "evil"),
    ("wand", "magic"),
    ("quidditch", "broomstick"),
    ("hermione", "book"),
]

for w1, w2 in pairs:
    sim = model.compute_similarity(w1, w2)
    print(f"Similarity between '{w1}' and '{w2}': {sim}")

# Step 5: Closest Words
print("\n--- Closest Words ---")
query_words = ["harry", "magic", "wand", "slytherin", "owl"]
for word in query_words:
    closest = model.get_closest_words(word, n=5)
    print(f"Words closest to '{word}': {closest}")
