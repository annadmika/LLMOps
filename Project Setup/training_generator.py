import pandas as pd
import random
import ast
import numpy as np

# --- Configuration (Set to 3000 for final run) ---
INPUT_CSV_FILE = 'tarot-images_update.csv'
OUTPUT_CSV_FILE = 'tarot_training_data.csv'
NUM_ROWS_TO_GENERATE = 5 # Change this to 5 for quick testing

def safe_literal_eval(val):
    """Safely converts a list-like string to a Python list, handling NaN and errors."""
    if pd.isna(val):
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return [val.strip()] if isinstance(val, str) else []

def split_mythical_spiritual(val):
    """Splits the Mythical/Spiritual string by '.' and cleans up the resulting list."""
    if pd.isna(val):
        return []
    # Replace non-standard separator 'Ñ' (used in one card) with comma for safety
    val = str(val).replace('Ñ', ',')
    # Split by period and filter out empty or whitespace-only strings
    parts = [part.strip() for part in val.split('.') if part.strip()]
    return parts

def get_keywords_from_question(question):
    """Simple tokenizer for filtering relevance."""
    # Filter common, low-value words
    stopwords = ['i', 'my', 'the', 'a', 'an', 'is', 'am', 'to', 'for', 'about', 'what', 'how', 'do', 'will', 'of', 'me', 'this', 'situation']
    # Extract relevant nouns/verbs, 4 letters or longer
    return [word for word in question.lower().split() if len(word) >= 4 and word not in stopwords]

def generate_tarot_reading(df):
    """Generates a single prompt and its corresponding structured, mystical response."""
    
    # --- A. Card Selection and Spread Assignment ---
    
    pulled_cards = df.sample(n=3, replace=False).reset_index(drop=True)
    
    # Define the spread positions
    spread = ['Situation', 'Action/Advice', 'Outcome/Result']
    
    # Randomly shuffle card assignments to positions
    reading = []
    card_indices = list(range(3))
    random.shuffle(card_indices)
    
    # The 'Outcome/Result' card will be the Key Card (index 2 in the spread list)
    key_card_position_index = card_indices.index(2) # Find which of the 3 pulled cards is assigned to the Outcome position (index 2)
    key_card_data = pulled_cards.loc[key_card_position_index]
    
    for i in range(3):
        card = pulled_cards.loc[i]
        position = spread[i]
        orientation = random.choice(['upright', 'reversed'])
        
        reading.append({
            'card': card,
            'position': position,
            'orientation': orientation,
            'is_key': (i == key_card_position_index)
        })

    # --- B. Build the PROMPT ---
    
    # Select question from the KEY CARD (The Outcome card)
    key_card_questions = key_card_data['Questions to Ask']
    user_question = random.choice(key_card_questions) if key_card_questions else "my current path in life"
    
    # Format card names with their orientation and position
    prompt_card_list = []
    for item in reading:
        name = item['card']['name']
        orientation = item['orientation']
        position = item['position']
        prompt_card_list.append(f"**{name}** ({orientation} in the **{position}** position)")

    prompt_cards_str = ", ".join(prompt_card_list)
    
    prompt = (
        f"The user is seeking guidance on: \"{user_question}\". "
        f"The user pulled the following three cards in a Situation-Action-Outcome spread: {prompt_cards_str}. "
        f"Perform a mystical, detailed tarot reading, relating each card's meaning to its spread position and the user's question."
    )
    
    # --- C. Build the RESPONSE (The Mystical Synthesis) ---
    
    response_parts = []
    
    # --- Greeting and Thesis Statement ---
    response_parts.append(
        f"Hark! I, the all-seeing mystic, observe the flow of destiny concerning your query on **\"{user_question}\"**. The veil thins to reveal a potent three-card narrative:"
    )
    
    # --- Card-by-Card Narrative ---
    for item in reading:
        card = item['card']
        orientation = item['orientation']
        position = item['position']
        name = card['name']
        arcana = card['arcana']
        
        # Determine the primary meaning to focus on
        meanings_list = card['meanings.light'] if orientation == 'upright' else card['meanings.shadow']
        primary_meaning = random.choice(meanings_list) if meanings_list else "an unfolding mystery."

        # Esoteric Enhancements (Numerology and Mythical/Spiritual)
        esoteric_part = ""
        if arcana == 'Major Arcana':
            # 1. Numerology
            if pd.notna(card['Numerology']) and card['Numerology'].strip():
                esoteric_part += f" This card vibrates with the divine numerology of the value **{card['Numerology']}**."
            
            # 2. Mythical/Spiritual (Random selection from the new list)
            if card['Mythical/Spiritual']:
                spiritual_concept = random.choice(card['Mythical/Spiritual'])
                esoteric_part += f" The spiritual energy of **{spiritual_concept}** is typically linked to this card."
        
        # Position-Specific Interpretation
        interpretation = f"As the **{position}** card, the **{name}** ({orientation}) brings a strong message: *'{primary_meaning}'*.{esoteric_part}"
        
        # Add a unique starter phrase based on the position
        if position == 'Situation':
            segment = f"\n\n**The Situation:** This reading begins with the **{name}**. This card describes the core dynamic you currently face. {interpretation}"
        elif position == 'Action/Advice':
            segment = f"\n\n**The Path to Take:** Next, the **{name}** offers the vital counsel on how to proceed. {interpretation}"
        elif position == 'Outcome/Result':
            segment = f"\n\n**The Destiny Revealed:** Finally, the **{name}** stands as the ultimate signifier for the answer to your question. {interpretation}"

        response_parts.append(segment)

    # --- Final Summary Logic (Higher Relevancy) ---
    
    # 1. Fortune Telling (from the KEY CARD, filtered by question)
    question_keywords = get_keywords_from_question(user_question)
    fortune_t = key_card_data['fortune_telling']
    
    # Filter for relevance
    relevant_fortunes = [ft for ft in fortune_t if any(keyword in ft.lower() for keyword in question_keywords)]
    
    # Choose from the relevant list if possible, otherwise use the general list
    fortune_t_source = relevant_fortunes if relevant_fortunes else fortune_t
    
    final_advice = "The threads of fate weave a simple warning: caution is advised."
    if fortune_t_source:
        ft_item = random.choice(fortune_t_source).replace('.', '').strip() 
        final_advice = f"**{ft_item}**"
    
    # 2. Affirmation (from the first non-empty affirmation among all 3 cards)
    affirmation_text = ""
    for item in reading:
        affirmation_candidate = item['card']['Affirmation']
        if pd.notna(affirmation_candidate) and affirmation_candidate.strip():
            affirmation_text = affirmation_candidate.strip().strip('"').strip()
            break 
            
    # 3. Construct the final message
    final_reading_summary = f"The oracle's final insight on your question: {final_advice}"
    if affirmation_text:
        final_reading_summary += f". During your journey, I advise you to repeat this sacred affirmation: **'{affirmation_text}'**"

    response_parts.append(
        f"\n\n**{final_reading_summary}**. Remember, the cards guide, but the power to choose is always your own."
    )
    
    return prompt, "".join(response_parts)

# ====================================================================
# --- MAIN EXECUTION BLOCK ---
# ====================================================================

if __name__ == "__main__":
    
    try:
        # Load the user-provided CSV
        df = pd.read_csv(INPUT_CSV_FILE)
    except FileNotFoundError:
        print(f"Error: The input file '{INPUT_CSV_FILE}' was not found. Ensure 'tarot-images.csv' is in the same directory.")
        exit()

    print(f"Loaded {INPUT_CSV_FILE}. Starting ADVANCED generation of {NUM_ROWS_TO_GENERATE} training examples...")

    # Preprocessing: Convert list-like strings to actual lists
    list_cols = [
        'fortune_telling', 'keywords', 'Questions to Ask', 
        'meanings.light', 'meanings.shadow'
    ]
    for col in list_cols:
        df[col] = df[col].apply(safe_literal_eval)
        
    # Preprocessing: Handle the Mythical/Spiritual column (NEWLY CORRECTED)
    df['Mythical/Spiritual'] = df['Mythical/Spiritual'].apply(split_mythical_spiritual)

    # Generate the dataset
    training_data = []
    for i in range(NUM_ROWS_TO_GENERATE):
        if (i + 1) % 500 == 0:
            print(f"  ... Generated {i + 1} rows...")
        prompt, response = generate_tarot_reading(df)
        training_data.append({'prompt': prompt, 'response': response})

    # Create the final DataFrame and save
    training_df = pd.DataFrame(training_data)
    training_df.to_csv(OUTPUT_CSV_FILE, index=False)

    print("-" * 50)
    print(f"✅ SUCCESSFULLY GENERATED {NUM_ROWS_TO_GENERATE} advanced training examples and saved to '{OUTPUT_CSV_FILE}'.")