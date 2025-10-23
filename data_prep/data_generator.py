import pandas as pd
import random
import ast
import numpy as np

# --- Configuration ---
INPUT_CSV_FILE = 'tarot-images.csv' 
OUTPUT_CSV_FILE = 'tarot_training_data.csv'
NUM_ROWS_TO_GENERATE = 5000 # Set this to 3000 for your final dataset. Change to 5 for quick testing.

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
    val = str(val).replace('Ñ', ',') # Handle the non-standard separator
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
    
    # --- 1. RANDOMIZATION PHRASES ---
    
    # Prompt Framing (Introduction)
    prompt_intros = [
        "The user is seeking guidance on:",
        "The user has posed the question:",
        "The user needs clarity on their path regarding:",
        "The query at hand is:"
    ]
    
    # Prompt Framing (Card Introduction)
    card_intros = [
        "The user pulled the following three cards in a Situation-Action-Outcome spread:",
        "The three cards drawn for the Situation-Action-Outcome spread are:",
        "The spread reveals these three cards in the order of Situation, Action, and Outcome:",
        "Here are the three cards pulled for this reading:"
    ]
    
    # Prompt Framing (Conclusion)
    prompt_conclusions = [
        "Perform a mystical, detailed tarot reading, relating each card's meaning to its spread position and the user's question.",
        "Provide a comprehensive reading, synthesizing the cards' meanings to address the user's query.",
        "Generate a detailed tarot interpretation, linking the cards to their spread roles and the user's focus.",
        "Deliver the full, narrative tarot reading based on the provided cards and question."
    ]
    
    # Response Framing (Greeting)
    response_openings = [
        f"Hark! I, the all-seeing mystic, observe the flow of destiny concerning your query on \"{{user_question}}\". The veil thins to reveal a potent three-card narrative:",
        f"Greetings. I have drawn the cards to illuminate your question about \"{{user_question}}\". The divine energies present a clear path:",
        f"Hello. Your request for guidance on \"{{user_question}}\" has been received. The cosmic spread reveals three powerful forces at play:",
        f"I see that you seek clarity on \"{{user_question}}\". The following reading will unveil the path forward:"
    ]
    
    # Varied Segment Titles (Response)
    title_options = {
        'Situation': ['The Situation', 'The Core Challenge', 'The Present Moment', 'The Current Reality', 'The Dynamic at Play'],
        'Action/Advice': ['The Path to Take', 'The Direction', 'The Way Forward', 'The Counsel Offered', 'The Key Action', 'The Guidance'],
        'Outcome/Result': ['The Destiny Revealed', 'The Final Outcome', 'The Potential Result', 'The Destiny Unveiled', 'The Ultimate Signifier']
    }
    
    # Image Description Phrases (Response)
    img_desc_phrases = [
        "Visually, this card is known for {img_desc}.",
        "The traditional imagery of this card depicts {img_desc}.",
        "This card is typically represented by {img_desc}.",
        "The scene on this card shows {img_desc}."
    ]
    
    # --- A. Card Selection and Assignment (Fixed Order) ---
    
    pulled_cards = df.sample(n=3, replace=False).reset_index(drop=True)
    
    # Define the fixed spread positions
    spread = ['Situation', 'Action/Advice', 'Outcome/Result']
    reading = []
    
    # The 'Outcome/Result' card (index 2 in the final reading list) is the Key Card
    key_card_data = pulled_cards.loc[2]
    
    for i in range(3):
        card = pulled_cards.loc[i]
        position = spread[i]
        orientation = random.choice(['upright', 'reversed'])
        
        reading.append({
            'card': card,
            'position': position,
            'orientation': orientation,
            'is_key': (i == 2)
        })

    # --- B. Build the PROMPT (Randomized Framing) ---
    
    # Select question from the KEY CARD (The Outcome card)
    key_card_questions = key_card_data['Questions to Ask']
    user_question = random.choice(key_card_questions) if key_card_questions else "my current path in life"
    
    # Format card names with their orientation and a varied position description (PROMPT RANDOMIZATION)
    prompt_card_list = []
    for item in reading:
        name = item['card']['name']
        orientation = item['orientation']
        position = item['position']
        
        # Vary the prompt's position description
        if position == 'Action/Advice':
            simple_name = random.choice(['Action', 'Advice', 'Guidance'])
            phrase = random.choice([f"as the {simple_name} card", f"in the {simple_name} position"])
        elif position == 'Outcome/Result':
            simple_name = random.choice(['Outcome', 'Result', 'Destiny'])
            phrase = random.choice([f"as the {simple_name} card", f"in the {simple_name} position"])
        else:
            simple_name = random.choice(['Situation', 'Challenge', 'Dynamic'])
            phrase = random.choice([f"as the {simple_name} card", f"in the {simple_name} position"])

        prompt_card_list.append(f"{name} ({orientation} {phrase})")

    # Final Prompt Construction
    prompt = (
        f"{random.choice(prompt_intros)} \"{user_question}\". "
        f"{random.choice(card_intros)} {prompt_card_list[0]}, {prompt_card_list[1]}, and {prompt_card_list[2]}. "
        f"{random.choice(prompt_conclusions)}"
    )
    
    # --- C. Build the RESPONSE (Varied Structure and Content - NO MARKDOWN) ---
    
    response_parts = []
    
    # 1. Greeting (Randomized)
    response_parts.append(random.choice(response_openings).format(user_question=user_question))
    
    # 2. Card-by-Card Narrative (Order is FIXED: Situation -> Action -> Outcome)
    for item in reading: 
        card = item['card']
        orientation = item['orientation']
        position = item['position']
        name = card['name']
        arcana = card['arcana']
        
        # --- BUILD CARD INFORMATION SEGMENTS ---
        
        # Segment 1: Card and Position Introduction
        random_title = random.choice(title_options[position])
        card_intro = f"\n\n{random_title}: The card drawn is the {name} ({orientation})."

        # Segment 2: Image Description (70% chance of inclusion)
        img_desc = card.get('img_description')
        img_segment = ""
        if img_desc is not None and pd.notna(img_desc) and img_desc.strip():
            if random.random() < 0.7:
                 # Select a random introductory phrase and remove the final period for cleaner concatenation
                 phrase = random.choice(img_desc_phrases).format(img_desc=img_desc.strip()).strip('.')
                 img_segment = f" {phrase}." # Ensure a period ends the sentence
        
        # Segment 3: Esoteric Details (Numerology/Mythicality - Random Inclusion)
        esoteric_segment = ""
        if arcana == 'Major Arcana':
            if random.random() < 0.5 and pd.notna(card['Numerology']) and card['Numerology'].strip():
                esoteric_segment += f" This card vibrates with the divine numerology of the value {card['Numerology']}."
            if random.random() < 0.7 and card['Mythical/Spiritual']:
                spiritual_concept = random.choice(card['Mythical/Spiritual'])
                esoteric_segment += f" The spiritual energy of {spiritual_concept} is typically linked to this card."
        
        # Segment 4: Core Meaning and Context
        meanings_list = card['meanings.light'] if orientation == 'upright' else card['meanings.shadow']
        primary_meaning = random.choice(meanings_list) if meanings_list else "an unfolding mystery."
        meaning_segment = f" As the {position} card, it brings a strong message: '{primary_meaning}'."

        # Combine all segments in the desired order
        segment = f"{card_intro}{img_segment}{esoteric_segment}{meaning_segment}"

        # Add unique starter phrase based on the position (for narrative flow)
        if position == 'Situation':
            final_segment = f"{segment} This describes the core dynamic you currently face."
        elif position == 'Action/Advice':
            final_segment = f"{segment} This offers the vital counsel on how to proceed."
        elif position == 'Outcome/Result':
            final_segment = f"{segment} This stands as the ultimate signifier for the answer to your question."
            
        response_parts.append(final_segment)

    # 3. Final Summary Logic (Higher Relevancy)
    
    # Fortune Telling (from the KEY CARD, filtered by question)
    question_keywords = get_keywords_from_question(user_question)
    fortune_t = key_card_data['fortune_telling']
    
    relevant_fortunes = [ft for ft in fortune_t if any(keyword in ft.lower() for keyword in question_keywords)]
    fortune_t_source = relevant_fortunes if relevant_fortunes else fortune_t
    
    final_advice = "The threads of fate weave a simple warning: caution is advised."
    if fortune_t_source:
        ft_item = random.choice(fortune_t_source).replace('.', '').strip() 
        final_advice = f"{ft_item}"
    
    # Affirmation (from the first non-empty affirmation among all 3 cards)
    affirmation_text = ""
    for item in reading:
        affirmation_candidate = item['card']['Affirmation']
        if pd.notna(affirmation_candidate) and affirmation_candidate.strip():
            affirmation_text = affirmation_candidate.strip().strip('"').strip()
            break 
            
    # CONSTRUCTING THE FINAL CLOSING (STRUCTURAL VARIATION)
    
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
        affirmation_part = f"During your journey, I advise you to repeat this sacred affirmation: '{affirmation_text}'"
    else:
        affirmation_part = ""
        
    final_reading_summary = closing_template.format(affirmation_part=affirmation_part).strip()

    response_parts.append(
        f"\n\n{final_reading_summary}"
    )
    
    return prompt, "".join(response_parts)

# ====================================================================
# --- MAIN EXECUTION BLOCK ---
# ====================================================================

if __name__ == "__main__":
    
    try:
        # FIX: Using latin-1 encoding to prevent UnicodeDecodeError
        df = pd.read_csv(INPUT_CSV_FILE, encoding='latin-1')
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