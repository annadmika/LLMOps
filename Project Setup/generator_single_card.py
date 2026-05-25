import pandas as pd
import random
import ast
import numpy as np

# --- Configuration ---
INPUT_CSV_FILE = 'tarot-images.csv' 
OUTPUT_CSV_FILE = 'tarot_training_data_SINGLE.csv'
NUM_ROWS_TO_GENERATE = 10 # Set this to 3000 for your final dataset. Change to 5 for quick testing.

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
    """Simple tokenizer for filtering relevance. (Kept for final advice/fortune-telling relevancy)."""
    stopwords = ['i', 'my', 'the', 'a', 'an', 'is', 'am', 'to', 'for', 'about', 'what', 'how', 'do', 'will', 'of', 'me', 'this', 'situation']
    return [word for word in question.lower().split() if len(word) >= 4 and word not in stopwords]

def generate_tarot_reading(df):
    """Generates a single prompt and its corresponding structured, mystical response (Single Card)."""
    
    # --- 1. RANDOMIZATION PHRASES ---
    
    # Prompt Framing (Introduction)
    prompt_intros = [
        "The user is seeking guidance on:",
        "The user has posed the question:",
        "The user needs clarity on their path regarding:",
        "The query at hand is:"
    ]
    
    # Prompt Framing (Card Introduction - Single Card)
    card_intros = [
        "The card drawn for this reading is:",
        "The card pulled is:",
        "The user drew the card:",
        "The deck revealed a single card:"
    ]
    
    # Prompt Framing (Conclusion)
    prompt_conclusions = [
        "Provide a detailed tarot reading in response to the user's query.",
        "Generate a comprehensive interpretation based on the card and the question.",
        "Deliver the full, narrative tarot reading based on the card and question.",
        "Offer a mystical interpretation of the card as it relates to the user's question."
    ]
    
    # Response Framing (Greeting)
    response_openings = [
        f"Hark! I, the all-seeing mystic, observe the flow of destiny concerning your query \"{{user_question}}\". The veil thins to reveal:",
        f"Greetings, I see that the deck has illuminated an answer to your question \"{{user_question}}\". The divine energies present:",
        f"Hello, the oracle has received your request for guidance on \"{{user_question}}\". The cosmic message is:",
        f"I see that you seek clarity on \"{{user_question}}\". The reading unveils:"
    ]
    
    # Image Description Phrases (Response)
    img_desc_phrases = [
        "Visually, this card is known for {img_desc}.",
        "The traditional imagery of this card depicts {img_desc}.",
        "This card is typically represented by {img_desc}.",
        "The scene on this card shows {img_desc}."
    ]
    
    # --- A. Card Selection and Assignment ---
    
    # Select only ONE card (which is the Key Card)
    pulled_card = df.sample(n=1).iloc[0]
    orientation = random.choice(['upright', 'reversed'])
    
    # --- B. Build the PROMPT (Single Card Logic) ---
    
    # Select question from the pulled card
    key_card_questions = pulled_card['Questions to Ask']
    user_question = random.choice(key_card_questions) if key_card_questions else "my current path in life"
    
    # Format card name for prompt
    prompt_card_name = f"{pulled_card['name']} ({orientation})"

    # Final Prompt Construction
    prompt = (
        f"{random.choice(prompt_intros)} \"{user_question}\". "
        f"{random.choice(card_intros)} {prompt_card_name}. "
        f"{random.choice(prompt_conclusions)}"
    )
    
    # --- C. Build the RESPONSE (Single Card Synthesis - NO MARKDOWN) ---
    
    response_parts = []
    
    # 1. Greeting (Randomized)
    response_parts.append(random.choice(response_openings).format(user_question=user_question))
    
    # --- Card Narrative ---
    
    card = pulled_card
    name = card['name']
    arcana = card['arcana']
    
    # Determine the primary meaning
    meanings_list = card['meanings.light'] if orientation == 'upright' else card['meanings.shadow']
    primary_meaning = random.choice(meanings_list) if meanings_list else "an unfolding mystery."

    # Esoteric Enhancements (Numerology and Mythical/Spiritual - Random Inclusion)
    esoteric_part = ""
    if arcana == 'Major Arcana':
        if random.random() < 0.5 and pd.notna(card['Numerology']) and card['Numerology'].strip():
            esoteric_part += f" This card vibrates with the divine numerology of the value {card['Numerology']}."
        if random.random() < 0.7 and card['Mythical/Spiritual']:
            spiritual_concept = random.choice(card['Mythical/Spiritual'])
            esoteric_part += f" The spiritual energy of {spiritual_concept} is typically linked to this card."

    # Image Description Sentence (Random Inclusion)
    img_desc = card.get('img_description')
    img_segment = ""
    if img_desc is not None and pd.notna(img_desc) and img_desc.strip():
        if random.random() < 0.7:
             phrase = random.choice(img_desc_phrases).format(img_desc=img_desc.strip()).strip('.')
             img_segment = f"\n\n{phrase}."
    
    # Segment Construction
    card_segment = (
        f"\n\nThe card drawn is the {name} ({orientation})."
        f"{img_segment}" # Image description insertion
        f"{esoteric_part}" # Esoteric details insertion
        f"\n\nThe core message for your query is: '{primary_meaning}'."
    )
    
    response_parts.append(card_segment)

    # --- Final Summary Logic (Relevancy Focused) ---
    
    # Fortune Telling (from the KEY CARD, filtered by question)
    question_keywords = get_keywords_from_question(user_question)
    fortune_t = pulled_card['fortune_telling']
    
    relevant_fortunes = [ft for ft in fortune_t if any(keyword in ft.lower() for keyword in question_keywords)]
    fortune_t_source = relevant_fortunes if relevant_fortunes else fortune_t
    
    final_advice = "The threads of fate weave a simple warning: caution is advised."
    if fortune_t_source:
        ft_item = random.choice(fortune_t_source).replace('.', '').strip() 
        final_advice = f"{ft_item}"
    
    # Affirmation (if available)
    affirmation_text = ""
    affirmation_candidate = pulled_card['Affirmation']
    if pd.notna(affirmation_candidate) and affirmation_candidate.strip():
        affirmation_text = affirmation_candidate.strip().strip('"').strip()
            
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
    
    # Choose a closing template
    closing_template = random.choice([template_1, template_2]) # Using two templates for single-card simplicity
    
    # Format the affirmation part based on whether one was found
    if affirmation_text:
        affirmation_part = f"During your journey, the oracle advises you to repeat this sacred affirmation: '{affirmation_text}'"
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

    print(f"Loaded {INPUT_CSV_FILE}. Starting Single-Card generation of {NUM_ROWS_TO_GENERATE} training examples...")

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
    print(f"✅ SUCCESSFULLY GENERATED {NUM_ROWS_TO_GENERATE} single-card training examples and saved to '{OUTPUT_CSV_FILE}'.")
