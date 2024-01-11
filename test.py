def convert_string(input_string):
    # Create a translation dictionary
    translation_dict = {'D': '3', 'R': '1', 'G': '2'}

    # Use str.translate method to replace characters
    return input_string.translate(str.maketrans(translation_dict))

# Example usage:
original_string = "DRRGRRRRRDRGDRGRRGRRDDDDGRRDRRGDGGRDDGGDGDRRDRRRRRRDGRDDRGRDRRGDDDGRDGGRRGGGDDDRGGGGRDRGGRRGRRRGDRRG"
converted_string = convert_string(original_string)
print(converted_string)
