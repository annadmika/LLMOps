import pandas as pd
import random
import ast
import numpy as np

# --- Configuration (Set to 3000 for final run) ---
INPUT_CSV_FILE = 'tarot-images.csv' # Assuming you're using the original name
OUTPUT_CSV_FILE = 'tarot_training_data.csv'
NUM_ROWS_TO_GENERATE = 3000 # Change this to 5 for quick testing

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
    val = str(val).replace('Ñ', ',')
    parts = [part.strip() for part in val.split('.') if part.strip()]
    return parts

def get_keywords_from_question(question):
    """Simple tokenizer for filtering relevance."""
    stopwords = ['i', 'my', 'the', 'a', 'an', 'is', 'am', 'to', 'for', 'about', 'what', 'how', 'do', 'will', 'of', 'me', 'this', 'situation']
    return [word for word in question.lower().split() if len(word) >= 4 and word not in stopwords]

def generate_tarot_reading(df):
    """Generates a single prompt and its corresponding structured, mystical response."""
    
    # --- 1. RANDOMIZATION PHRASES ---
    openings = [
        f"Hark! I, the all-seeing mystic, observe the flow of destiny concerning your query on **\"{{user_question}}\"**. The veil thins to reveal a potent three-card narrative:",
        f"Greetings. I have drawn the cards to illuminate your question about **\"{{user_question}}\"**. The divine energies present a clear path:",
        f"Hello. Your request for guidance on **\"{{user_question}}\"** has been received. The cosmic spread reveals three powerful forces at play:",
        f"I see that you seek clarity on **\"{{user_question}}\"**. The following reading will unveil the path forward:"
    ]
    
    # --- Varied Segment Titles ---
    title_options = {
        'Situation': ['The Situation', 'The Core Challenge', 'The Present Moment', 'The Current Reality', 'The Dynamic at Play'],
        'Action/Advice': ['The Path to Take', 'The Direction', 'The Way Forward', 'The Counsel Offered', 'The Key Action', 'The Guidance'],
        'Outcome/Result': ['The Destiny Revealed', 'The Final Outcome', 'The Potential Result', 'The Destiny Unveiled', 'The Ultimate Signifier']
    }
    
    # --- A. Card Selection and Spread Assignment ---
    
    pulled_cards = df.sample(n=3, replace=False).reset_index(drop=True)
    
    # Define the spread positions for the final, FIXED order
    spread = ['Situation', 'Action/Advice', 'Outcome/Result']
    
    # Assign the cards to the fixed positions
    reading = []
    
    # The 'Outcome/Result' card will be the Key Card (index 2)
    key_card_data = pulled_cards.loc[2]
    
    for i in range(3):
        card = pulled_cards.loc[i]
        position = spread[i]
        orientation = random.choice(['upright', 'reversed'])
        
        reading.append({
            'card': card,
            'position': position,
            'orientation': orientation,
            'is_key': (i == 2) # Outcome is always the Key Card
        })
        
    # NOTE: The random shuffling of the *interpretation order* is removed.
    # The 'reading' list is now [Situation, Action/Advice, Outcome/Result].

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
        f"The user pulled the following three cards in a Situation-Action-Outcome spread: {prompt_card_list[0]}, {prompt_card_list[1]}, and {prompt_card_list[2]}. "
        f"Perform a mystical, detailed tarot reading, relating each card's meaning to its spread position and the user's question."
    )
    
    # --- C. Build the RESPONSE (The Mystical Synthesis) ---
    
    response_parts = []
    
    # 1. Greeting (Randomized)
    response_parts.append(random.choice(openings).format(user_question=user_question))
    
    # 2. Card-by-Card Narrative (Order is now FIXED by the 'reading' list)
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
            
            # 50% chance for Numerology
            if random.random() < 0.5 and pd.notna(card['Numerology']) and card['Numerology'].strip():
                esoteric_part += f" This card vibrates with the divine numerology of the value **{card['Numerology']}**."
            
            # 70% chance for Mythical/Spiritual
            if random.random() < 0.7 and card['Mythical/Spiritual']:
                spiritual_concept = random.choice(card['Mythical/Spiritual'])
                esoteric_part += f" The spiritual energy of **{spiritual_concept}** is typically linked to this card."
        
        # Position-Specific Interpretation
        # Get Randomized Title
        random_title = random.choice(title_options[position])
        
        interpretation = f"As the **{position}** card, the **{name}** ({orientation}) brings a strong message: *'{primary_meaning}'*.{esoteric_part}"
        
        # Segment construction using the randomized title
        if position == 'Situation':
            segment = f"\n\n**{random_title}:** This card describes the core dynamic you currently face. {interpretation}"
        elif position == 'Action/Advice':
            segment = f"\n\n**{random_title}:** This card offers the vital counsel on how to proceed. {interpretation}"
        elif position == 'Outcome/Result':
            segment = f"\n\n**{random_title}:** This card stands as the ultimate signifier for the answer to your question. {interpretation}"

        response_parts.append(segment)

    # --- Final Summary Logic (Higher Relevancy) ---
    
    # 1. Fortune Telling (from the KEY CARD, filtered by question)
    question_keywords = get_keywords_from_question(user_question)
    fortune_t = key_card_data['fortune_telling']
    
    relevant_fortunes = [ft for ft in fortune_t if any(keyword in ft.lower() for keyword in question_keywords)]
    
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
            
    # 3. CONSTRUCTING THE FINAL CLOSING (STRUCTURAL VARIATION)
    
    # Template 1: Focus on advice first, then affirmation.
    template_1 = (
        f"The oracle's final insight on your question: {final_advice}. "
        f"{{affirmation_part}} "
        f"May the wisdom of the Arcana illuminate your next steps on your journey."
    )
    
    # Template 2: Focus on the cosmic/fate message, then advice, then affirmation.
    template_2 = (
        f"The cosmos confirms your reading with a final message: {final_advice}. "
        f"{{affirmation_part}} "
        f"Go forth, for the power to choose is always your own."
    )
    
    # Template 3: More direct, emphasizing empowerment.
    template_3 = (
        f"The final decree from the cards is clear: {final_advice}. "
        f"{{affirmation_part}} "
        f"Remember that the cards only guide; your own spirit holds the key to destiny."
    )
    
    # Choose a closing template
    closing_template = random.choice([template_1, template_2, template_3])
    
    # Format the affirmation part based on whether one was found
    if affirmation_text:
        affirmation_part = f"During your journey, I advise you to repeat this sacred affirmation: **'{affirmation_text}'**"
    else:
        affirmation_part = ""
        
    final_reading_summary = closing_template.format(affirmation_part=affirmation_part).strip()

    response_parts.append(
        f"\n\n**{final_reading_summary}**"
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
        print(f"Error: The input file '{INPUT_CSV_FILE}' was not found. Please ensure your data file is in the same directory.")
        exit()

    print(f"Loaded {INPUT_CSV_FILE}. Starting ADVANCED generation of {NUM_ROWS_TO_GENERATE} training examples...")

    # Preprocessing: Convert list-like strings to actual lists
    list_cols = [
        'fortune_telling', 'keywords', 'Questions to Ask', 
        'meanings.light', 'meanings.shadow'
    ]
    for col in list_cols:
        df[col] = df[col].apply(safe_literal_eval)
        
    # Preprocessing: Handle the Mythical/Spiritual column
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
    print("\n--- Example of First Row (Check FIXED order and randomization) ---")
    print("\n**Prompt:**\n", training_df.loc[0, 'prompt'])
    print("\n**Response:**\n", training_df.loc[0, 'response'])