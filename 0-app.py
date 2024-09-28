import streamlit as st
from src.module import QuizGenerator, InputsHandler, QuizProcessor


st.set_page_config("Quiz-Taker", page_icon="🎓")
DETAILS_DF_PATH = "classes_data_details.csv"
INPUTS_HANDLER = InputsHandler(DETAILS_DF_PATH)
SUPPORTED_CLASSES = INPUTS_HANDLER.get_supported_classes_list()
DIFFICULTY_LEVELS = ["Basic", "Intermediate", "Advance"]


# front-end
st.header("Quiz Taker 🎓")

col1, col2, col3, col4 = st.columns(4)
with col1:
    selected_class = st.selectbox("Class", options=SUPPORTED_CLASSES, index=0)
    selected_difficulty = st.selectbox("Difficulty", options=DIFFICULTY_LEVELS)
    
with col2:
    selected_subject = st.selectbox("Subject", options=INPUTS_HANDLER.get_selected_class_subjects(selected_class), index=0)
    num_questions = st.number_input("No. of Questions", value=5)
    
with col3:
    chapter_name = st.text_input("Chapter name")
    num_pages = st.number_input("Pages to use.", value=3)

with col4:
    topic = st.text_input("Topic")

generate_quiz_btn = st.button("Generate Quiz", type="primary", use_container_width=True)


# inputs handling backend
if generate_quiz_btn:
    if not topic:
        st.error("Please provide topic.")
        st.stop()
    
    with st.spinner("Generating.."):
        QUIZ_GENERATOR = QuizGenerator(selected_class, selected_subject, chapter_name, num_questions, selected_difficulty, topic, num_pages)
        quiz = QUIZ_GENERATOR.generate_quiz()
        
        # Store the quiz in session state
        st.session_state['quiz'] = quiz
        st.session_state['quiz_generated'] = True

# Check if quiz is already generated
if st.session_state.get('quiz_generated'):
    QUIZ_PROCESSOR = QuizProcessor(st.session_state['quiz'])
    quiz = QUIZ_PROCESSOR.get_formatted_quiz()

    # Display the quiz
    for question, radio_details in quiz:
        st.write(question)
        
        # Store answers in session state
        if radio_details['key'] not in st.session_state:
            st.session_state[radio_details['key']] = None

        st.radio(
            label=radio_details['label'],
            options=radio_details['options'],
            key=radio_details['key']
        )
    submit_btn = st.button("Submit", type="primary")

    # evaluating performance
    if submit_btn:
        if not QUIZ_PROCESSOR.all_questions_answered(st.session_state):
            st.error("Please answer all questions.")
        else:
            correct_count, wrong_details = QUIZ_PROCESSOR.evaluate_student_performance(st.session_state)
            
            # Display the result
            st.write(f"You got {correct_count} out of {num_questions}!")
            st.write(f"Correct choices: \n {wrong_details}")
