def process_text(text):
    """Preprocess and clean up text before sending it to the model."""
    text = text.replace("\n", " ").strip()
    # You can add further processing logic here
    return text
