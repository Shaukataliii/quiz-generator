import streamlit as st

# Sample MCQ data
response = [
    {
        'question': 'What is the relationship between force, inertia, and momentum?',
        'choices': ['Force causes a body to change its direction of motion.',
                    "Inertia opposes any changes in a body's state of rest or uniform motion.",
                    'Force and inertia are independent concepts.',
                    'Momentum is directly proportional to the force applied.'],
        'correct_choice': 1
    },
    {
        'question': "What does Newton's law of motion say about an object's motion?",
        'choices': ['An object at rest will stay at rest unless a force acts upon it.',
                    'An object in motion will always continue in that motion unless acted upon by a force.',
                    'The net force acting on an object is equal to its mass multiplied by its acceleration.',
                    'Both (a) and (c)'],
        'correct_choice': 3
    },
    {
        'question': 'What happens when you push down on a balloon?',
        'choices': ['The balloon expands due to the pressure of the air trapped inside.',
                    'The balloon shrinks because of the force of your hand pushing it.',
                    'The balloon stays the same size as there is no external force acting on it.',
                    'The balloon changes shape based on its inertia.'],
        'correct_choice': 1
    },
    {
        'question': "What is inertia and how does it relate to an object's movement?",
        'choices': ['Inertia is the tendency of an object to resist a change in its velocity.',
                    'Inertia is the force that keeps an object at rest.',
                    'Inertia is the ability to move without external help.',
                    'Inertia is the resistance of an object to acceleration.'],
        'correct_choice': 0
    },
    {
        'question': 'Why does a coin remain stationary when you pull the paper strip away?',
        'choices': ["The coin's inertia prevents it from moving.",
                    'The paper strip is heavier than the coin, so it moves first and then the coin follows.',
                    'The string tension stops the coin from moving.',
                    'The force of the hand pulling on the paper strip makes the coin move.'],
        'correct_choice': 0
    }
]

st.title("MCQ Quiz")

# To collect user answers
user_answers = []

# Display each question and its choices
for index, mcq in enumerate(response):
    q_no = index + 1
    st.write(f"Q{q_no}: {mcq['question']}")
    
    # Display radio buttons for each question
    st.radio(f"Select your answer for Question {q_no}:",
                mcq['choices'], key=f"q_{q_no}", index=None)
    

# Submit button
if st.button("Submit"):

    correct_count = 0
    wrong_details = []
    
    # Evaluate the user's answers
    try:
        for q_original_idx, mcq in enumerate(response):

            q_displayed_idx = q_original_idx + 1
            q_correct_idx = mcq['correct_choice']
            selected_answer = st.session_state[f"q_{q_displayed_idx}"]
            selected_answer_idx = mcq['choices'].index(selected_answer)

            if selected_answer_idx == q_correct_idx:
                correct_count += 1
            else:
                wrong_details.append(f"Q{q_displayed_idx}: {q_correct_idx + 1}")

        # Display the result
        st.write(f"You got {correct_count} out of {len(response)} correct!")
        st.write(f"Correct choices: \n {wrong_details}")

    except ValueError:
        st.error("Please answer all questions.")
    
