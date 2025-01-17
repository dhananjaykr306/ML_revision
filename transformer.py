# @Author: Dhananjay Kumar
# @Date: 17-01-2025
# @Last Modified by: Dhananjay Kumar
# @Last Modified time: 17-01-2025
# @Title: Python program to use a pre-trained Transformer model to classify text sentiment (positive, negative, neutral). Describe the steps you would take to load the model, prepare your dataset, and make predictions on new text inputs.



from groq import Groq
import os
def main():
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    review = input("Enter the review:")
    # Define the input for the model
    input_message = {
        "role": "user",
        "content": review
    }

    # Define the prompt for the model
    prompt = {
        "role": "assistant",
        "content": "What is the sentiment of the following review in one word? positive, negative, or neutral?"
    }

    # Define the chat completion
    chat_completion = client.chat.completions.create(
        messages=[prompt, input_message],
        model="llama-3.3-70b-versatile",
        stream=False,
    )

    # Print the model's response
    print("\n Review's sentiment: ",chat_completion.choices[0].message.content)

if __name__ == "__main__":
    main()
