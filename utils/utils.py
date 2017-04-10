import re

def clean_text(text):
	text = re.sub(r'[^\w ]', '', text)
	text = re.sub(r'(\s)+', r'\1', text)
	return text.lower()

def clean_texts(texts):
	return list(map(clean_text, texts))

	