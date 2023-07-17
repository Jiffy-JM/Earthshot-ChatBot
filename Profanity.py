from better_profanity import profanity

def is_text_offensive(text):
    return profanity.contains_profanity(text)

# Example usage:
user_input = input("Enter your text: ")
if is_text_offensive(user_input):
    print("Your input contains offensive language. Please refrain from using inappropriate words.")
else:
    print("Thank you for your input.")
