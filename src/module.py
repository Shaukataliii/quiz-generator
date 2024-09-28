import streamlit as st
from langchain_codebase.codebase import *
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from pdf2image import convert_from_path
import pytesseract
import ast


QUIZ_GENERATOR_PROMPT = """
You will receive the contents of a particular chapter of a student's book. Your job is to create {no_questions} multiple choice question answers using those contents and provide them in the form of python code where:
1- All are questions are inside a list.
2- Each question is a dictionary of values.
3- The keys of the question dictionary should be:
question (representing question as str)
choices (list of choices representing each choice as str)
correct_choice (the index of correct choice i.e. the index of correct choice form the choices list.)

Note: Strictly follow the rules and don't provide anything else than the asked. Only provide a clean list ready to process further.

contents: {book_content}
"""
PARAMS_FILEPATH = r'src\params.yaml'
BOOKS_VDB_PATH = "VDB-ALL-BOOKS"
BOOKS_DF_DIRNAME = "books_df_csvs"
CLASSES_DATA_FILEPATH = 'classes_data_details.csv'
CLASS_KEY = "class"
SUBJECT_KEY = "subject"
CONTENT_KEY = "content"
PAGE_NO_KEY = "page_no"


class DocumentProcessor:

    def extract_details_from_docs_as_df(self, docs, subject, book_class):
        book_data = {
            CLASS_KEY: [],
            SUBJECT_KEY: [],
            PAGE_NO_KEY: [],
            CONTENT_KEY: []
        }
        for doc in docs:
            page_no, content = self.extract_page_no_page_content_from_doc(doc)
            book_data[CLASS_KEY].append(book_class)
            book_data[SUBJECT_KEY].append(subject)
            book_data[PAGE_NO_KEY].append(page_no)
            book_data[CONTENT_KEY].append(content)

        book_data_df = pd.DataFrame(book_data)
        self.check_for_missing_content_val(book_data_df)
        
        return book_data_df

    def extract_page_no_page_content_from_doc(self, doc):
        page_no = doc.metadata['page']
        page_content = doc.page_content
        return (page_no, page_content)

    def check_for_missing_content_val(self, df: pd.DataFrame):
        """Checks how many rows don't have content information. If all rows don't have, then raises exception."""
        num_rows = df.shape[0]
        pages_with_no_content = 0

        for _, row in df.iterrows():
            content = row[CONTENT_KEY]
            
            if not content:
                pages_with_no_content += 1
        
        print(f"\nTotal pages: {num_rows}")
        print(f"Pages with no content: {pages_with_no_content}")

        if (pages_with_no_content == num_rows):
            raise Exception("Content not present on any page.")


class BookProcessor:

    def process_book_pdf_and_save_df(self, book_pdf_path: str, contain_images: bool = False):
        """Extracts book_pdf_name, subject and book_class from the provided path. Uses them to extracts details as DataFrame and saves them using book_pdf_name and add the book availability details to classes_data_details."""
        self.set_book_path_detail_vars(book_pdf_path)
        df = self.extract_details_from_book_pdf_path_as_df(contain_images)
        self.save_book_df(df)
        BOOKS_DETAILS_MANAGER.add_book_details_to_details_csv(self.book_class, self.subject)
    
    def extract_details_from_book_pdf_path_as_df(self, contain_images: bool = False):
        if is_valid_pdf_path(self.book_pdf_path):
            docs = load_single_pdf(self.book_pdf_path, contain_images)
            df = DOC_PROCESSOR.extract_details_from_docs_as_df(docs, self.subject, self.book_class)
            return df


    def use_ocr_to_process_book_pdf_and_save_df(self, book_pdf_path: str):
        """Uses OCR to process the pdf.
        Extracts book_pdf_name, subject and book_class from the provided path. Uses them to process extracts details as DataFrame and saves them using book_pdf_name."""
        self.set_book_path_detail_vars(book_pdf_path)
        df = self.use_ocr_to_extract_details_from_book_pdf_path_as_df()
        self.save_book_df(df)
        BOOKS_DETAILS_MANAGER.add_book_details_to_details_csv(self.book_class, self.subject)

    def use_ocr_to_extract_details_from_book_pdf_path_as_df(self):
        if is_valid_pdf_path(self.book_pdf_path):
            docs = OCR_USER.convert_pdf_path_to_docs(self.book_pdf_path, self.book_pdf_name)
            df = DOC_PROCESSOR.extract_details_from_docs_as_df(docs, self.subject, self.book_class)
            return df
        
    
    def set_book_path_detail_vars(self, book_pdf_path: str):
        self.book_pdf_path = book_pdf_path
        self.book_pdf_name, self.subject, self.book_class = BOOK_PATH_PROCESSOR.extract_filename_subject_class_from_book_pdffile_path(book_pdf_path)
        self.book_name = (self.subject + " " + self.book_class)


    def save_book_df(self, df: pd.DataFrame):
        if not self.csv_files_dir_exist():
            os.makedirs(BOOKS_DF_DIRNAME, exist_ok=True)

        book_save_path = os.path.join(BOOKS_DF_DIRNAME, self.book_name + ".csv")
        df.to_csv(book_save_path, index=False)
        print(f"Book saved as: {book_save_path}")
        
    def csv_files_dir_exist(self):
        if os.path.exists(BOOKS_DF_DIRNAME):
            return True
        else:
            return False


class BookPathProcessor:
    def extract_filename_subject_class_from_book_pdffile_path(self, book_pdf_path: str):
        self.book_pdf_path = book_pdf_path

        self.book_pdf_name = self.extract_book_pdffile_name_from_path()
        subject, book_class = self.extract_subject_and_class_from_book_pdffile_name()
        return self.book_pdf_name, subject, book_class

    def extract_book_pdffile_name_from_path(self):
        book_pdf_name = os.path.basename(self.book_pdf_path)
        return book_pdf_name

    def extract_subject_and_class_from_book_pdffile_name(self):
        """Book pdffile name is supposed to be like: 'Physics 9.pdf', so it separates using space and return the values. The class name should be isnumeric."""
        book_name = self.book_pdf_name.rsplit(".", 1)[0]
        if not ' ' in book_name:
            raise Exception(f"Book name doesn't contain any space. It is: {book_name}")
        
        if not self.name_ends_with_numeric_char(book_name):
            raise Exception(f"Book name doesn't class i.e. it doesn't end with a numeric character.")
        
        subject, book_class = book_name.rsplit(" ", 1)
        return (subject, book_class)
    
    def name_ends_with_numeric_char(self, name: str):
        end_char = name[-1]
        return True if end_char.isnumeric() else False


class BookDetailsManager:
    """Adds the provided class and subject info to the existing classes_details_df.
    """
    def __init__(self):
        pass

    def add_book_details_to_details_csv(self, class_no: str, subject: str):
        """Add the provided details into the classes details CSV file.
        """
        new_data = self._convert_details_to_df(class_no, subject)
        self._extend_classes_details_df(new_data)

    def _convert_details_to_df(self, class_no: int, subject: str) -> pd.DataFrame:
        """Converts class, subject into a DataFrame.
        """
        return pd.DataFrame({
            CLASS_KEY: [class_no],
            SUBJECT_KEY: [subject]
        })

    def _extend_classes_details_df(self, new_df: pd.DataFrame):
        """Appends the new data (df) to the existing details CSV."""
        old_df = UTILS.load_pandas_df(CLASSES_DATA_FILEPATH)
        updated_df = pd.concat([old_df, new_df], ignore_index=True)
        self._save_extended_df(updated_df)
        print("Book details added successfully.")

    def _save_extended_df(self, df: pd.DataFrame):
        """Saves the extended DataFrame."""
        df.to_csv(CLASSES_DATA_FILEPATH, index=False)


class OCRUser:
    def __init__(self):
        params = load_yaml_file(PARAMS_FILEPATH)
        self.poppler_path = params['poppler_path']

    def convert_pdf_path_to_docs(self, book_pdf_path: str, book_pdf_name: str):
        self.book_pdf_path = book_pdf_path
        self.book_pdf_name = book_pdf_name

        pages = self.convert_pdf_path_to_images()
        docs = self.convert_pdf_images_to_docs(pages)
        return docs

    def convert_pdf_path_to_images(self):
        images = convert_from_path(pdf_path = self.book_pdf_path, poppler_path = self.poppler_path)
        return images


    def convert_pdf_images_to_docs(self, pdf_images):
        docs = []
        for image_no, image in enumerate(pdf_images):
            image_no = (image_no + 1)
            image_doc = self.convert_pdf_image_to_doc(image, image_no)
            docs.append(image_doc)
        return docs

    def convert_pdf_image_to_doc(self, pdf_image: str, image_no: int):
        text = pytesseract.image_to_string(pdf_image)
        doc = Document(page_content=text)
        doc.metadata['page'] = image_no
        doc.metadata['source'] = self.book_pdf_name
        return doc


class InputsHandler:
    """
    Handles the inputs for:
    1- Front-end app using provided dataframe path and provides details i.e. available classes and available subjects depending on selections.
    2- For backend single book processing and saving its dataframe. 

    Provided dataframe should have the following structure:
    0: class_details,
    1: subject_details
    """
    def __init__(self, details_df_path: str):
        """
        Initializes dataframe. Raises exceptions if path is invalid or is not a csv file.
        """
        details_df = UTILS.load_pandas_df(details_df_path)
        self.details_df = details_df

    def get_supported_classes_list(self) -> list:
        """
        Returns list of classes available in the dataframe.
        """
        supported_classes = self.details_df[CLASS_KEY]
        return supported_classes.unique().tolist()

    def get_selected_class_subjects(self, selected_class: int) -> list:
        """
        From the dataframe, Returns a list of available subjects for the provided class.
        """
        subjects = self._get_class_df(selected_class)[SUBJECT_KEY]
        return subjects.unique().tolist()
    
    def take_evaluate_inputs_to_process_single_book(self):
        """Takes pdf_path, contain_images, use_ocr input parameters, evalutes them and on successful evaluation.
        returns (pdf_path: str, contain_images: bool, use_ocr: bool)."""
        self.pdf_path = input("Enter book pdf file path: ")
        if is_valid_pdf_path(self.pdf_path):
            return self._take_other_inputs_and_process_them()

    def evaluate_convert_val_to_bool(self, value: str):
        """The value can only be yes or no. Transforms to True if yes else False."""
        if value not in ['yes', 'no']:
            raise Exception(f"Invalid Value. Expected yes or no. Got: {value}")
        
        value = (value == 'yes')
        return value

    def _get_class_df(self, selected_class: str) -> pd.DataFrame:
        """
        Internal method to filter dataframe for the provided class.
        Raises ValueError if class not found.
        """
        class_df = self.details_df[self.details_df[CLASS_KEY]==selected_class]

        if class_df.empty:
            raise ValueError(f"Class: {selected_class} is not present in dataframe.")
        
        return class_df
    
    def _take_other_inputs_and_process_them(self):
        contain_images = input("Extract details from images (yes/no): ")
        use_ocr = input("Use OCR? (yes/no) ")
        sure = input("Are you sure? (yes/no) ")

        sure = self.evaluate_convert_val_to_bool(sure)
        use_ocr = self.evaluate_convert_val_to_bool(use_ocr)
        contain_images = self.evaluate_convert_val_to_bool(contain_images)

        if sure:
            return (self.pdf_path, contain_images, use_ocr)
        else:
            return self.handle_pdfpath_input()


class QuizGenerator:
    """
    Generates quizzes based on provided class, subject, and other parameters. 
    It interacts with the Vectorstore and LLM, retrieves relevant documents and generates quiz.
    """
    def __init__(self, selected_class: int, subject: str, chapter_name: str, num_questions: int, difficulty_level: str, topic: str, num_docs: int = 3):
        """
        Initializes QuizGenerator with the necessary parameters.

        :param selected_class: The class number (e.g., 10 for 10th grade).
        :param subject: The subject for which to generate the quiz.
        :param chapter_name: The subject chapter name.
        :param num_questions: The number of questions to generate in the quiz.
        :param difficulty_level: The difficulty level of the quiz (e.g., easy, medium, hard).
        :param topic: The topic or keyword for quiz generation.
        :param num_docs: The number of relevant documents to retrieve from the vector store (default is 3).
        """
        self.selected_class = selected_class
        self.subject = subject
        self.num_questions = num_questions
        self.difficulty_level = difficulty_level
        self.query = chapter_name + " " + topic
        self.num_docs = num_docs

        # self.llm = UTILS.load_ollama_llm()
        self.llm = get_google_llm()
        self.vectorstore = load_vectorstore(BOOKS_VDB_PATH)

    def generate_quiz(self) -> list:
        """
        Generates a quiz by fetching relevant documents, constructing a prompt, and querying the LLM.

        :return: The generated quiz as a list of questions.
        :raises ValueError: If no relevant documents are found or the LLM response is invalid.
        """
        try:
            docs = self._get_relevant_docs_as_string()
            if not docs:
                raise ValueError(f"No documents found for the given criteria.")

            prompt = self._get_formatted_prompt(docs)
            quiz = self._generate_quiz_from_prompt(prompt)
            return quiz
        
        except Exception as e:
            raise ValueError(f"Error generating quiz. {str(e)}")
        

    def _get_relevant_docs_as_string(self):
        """
        Fetches relevant documents from the vector store and combines them into a single string.

        :return: Combined text of relevant documents.
        :raises ValueError: If no documents are retrieved.
        """
        docs = self._fetch_relevant_docs_from_vectorstore()
        if not docs:
            raise ValueError(f"No documents found for the given criteria.")
        
        docs = [doc.page_content for doc in docs]
        docs = " ".join(docs)
        print(docs)
        return docs

    def _fetch_relevant_docs_from_vectorstore(self):
        """
        Performs a similarity search on the vector store using the class and subject as filters and query as query.

        :return: A list of documents retrieved from the vector store.
        :raises ValueError: If any search parameters are missing or invalid.
        """
        try:
            filters = {
                '$and' : [
                    {CLASS_KEY: {'$eq': self.selected_class}},
                    {SUBJECT_KEY: {'$eq': self.subject}}
                ]
            }
            similar_docs = self.vectorstore.similarity_search(self.query, k=self.num_docs, filter=filters)
            return similar_docs
        
        except Exception as e:
            raise ValueError(f"Error fetching documents from vector store: {str(e)}")


    def _get_formatted_prompt(self, docs: str):
        """
        Formats the prompt for the LLM by embedding the quiz parameters and document content.

        :param docs: A string containing the combined content of relevant documents.
        :return: The formatted prompt string.
        :raises ValueError: If the prompt template is invalid.
        """
        try:
            prompt = self._get_quiz_generator_prompt()
            prompt = prompt.format(no_questions=self.num_questions, difficulty_level=self.difficulty_level, book_content=docs)
            return prompt
        
        except Exception as e:
            raise ValueError(f"Invalid prompt template: {str(e)}")
        
    def _get_quiz_generator_prompt(self):
        """
        Returns the quiz generation prompt template.

        :return: A PromptTemplate object with the template for quiz generation.
        """
        prompt = PromptTemplate(
            input_variables = ['no_questions', 'difficulty_level', 'book_content'],
            template = QUIZ_GENERATOR_PROMPT
        )
        return prompt


    def _generate_quiz_from_prompt(self, prompt):
        """
        Sends the formatted prompt to the LLM and evaluates the response.

        :param prompt: The formatted prompt string.
        :return: A list of quiz questions.
        :raises ValueError: If the LLM response is malformed or contains invalid data.
        """
        try:
            response = self._get_response_from_llm(prompt)
            response = self._evaluate_generated_quiz(response)
            return response
        
        except Exception as e:
            raise ValueError(f"Error generating quiz from LLM response: {str(e)}")

    def _get_response_from_llm(self, prompt):
        """
        Sends a request to the LLM with the given prompt and returns its response.

        :param prompt: The prompt string to send to the LLM.
        :return: The raw response from the LLM.
        :raises RuntimeError: If the LLM invocation fails.
        """
        try:
            return self.llm.invoke(prompt)
        
        except Exception as e:
            raise RuntimeError(f"LLM invocation failed: {str(e)}")

    def _evaluate_generated_quiz(self, response: str):
        """
        Processes the LLM response, removing unwanted text and converting it to a Python object.

        :param response: The raw response from the LLM.
        :return: The evaluated response, typically a list of quiz questions.
        :raises ValueError: If the response cannot be parsed as a valid Python object.
        """
        try:
            response = self._remove_unwanted_chars_from_response(response)
            response = ast.literal_eval(response)
            return response
        
        except Exception as e:
            raise ValueError(f"Failed to evaluate generated quiz: {str(e)}")
    
    def _remove_unwanted_chars_from_response(self, response: str):
        """
        Removes 'python' and '```' form response and returns the response.
        """
        response = response.replace("python", '').replace("```", '')
        return response


class QuizProcessor:
    """
    Uses the generated quiz and processes it. Provides formatted strings to display the questions and evalute student performance.
    
    :param quiz:
        A list containing question dictionaries having keys:
        - question: the question (str)
        - options: list of options (list)
        - answer: index of correct option (int)
    """
    def __init__(self, quiz: list):
        """Initializes the provided quiz. Raises TypeError is quiz is not a list."""

        if not isinstance(quiz, list):
            raise TypeError("quiz must be a list.")
        
        self.quiz = quiz


    def get_formatted_quiz(self):
        """
        Formats the quiz into a list of questions to use in streamlit. Each question is formated into a list containing:
            - question text
            - dictionary to use for radio button containing keys:
                - label
                - options
                - key : The question key to use. This is to facilitate the evaluation process.

        :return: Quiz containing formatted questions (list).
        """
        quiz = []
        for index, mcq in enumerate(self.quiz):
            q_no = index + 1
            q_key = self._get_question_key(index)

            question = f"Q{q_no}: {mcq['question']}"
            radio = {
                'label': "Select your answer:", 
                'options': mcq['choices'], 
                'key': q_key
                }
            formatted_q = [question, radio]
            quiz.append(formatted_q)

        return quiz
    
    def evaluate_student_performance(self, session) -> tuple:
        """
        Evaluates student answers based on correct answers in quiz.

        :param session: The streamlit session state after user has answers the quiz.
        :return: A tuple containing:
            - Number of correct answers (int)
            - Details of the wrong answers in format:- "Q question_no: correct_choice, ...
        """
        correct_count = 0
        wrong_details = []
        for index, mcq in enumerate(self.quiz):
            q_no = index + 1
            q_key = self._get_question_key(index)
            correct_option_index = mcq['correct_choice']

            answer_text = session[q_key]
            answer_index_in_mcq = mcq['choices'].index(answer_text)

            if answer_index_in_mcq == correct_option_index:
                correct_count += 1
            else:
                detail = f"Q{q_no}: {correct_option_index + 1}"
                wrong_details.append(detail)

        return (correct_count, ", ".join(wrong_details))
        
        
    def all_questions_answered(self, session):
        """
        Checks if all the questions in the quiz are answered.
        For each question, checks if the answer is None which indicates unanswered question.

        :param session: The streamlit session state containing user answers.
        :return: True if all questions are answered else False.
        """
        for index in range(len(self.quiz)):
            q_key = self._get_question_key(index)
            answer_text = session[q_key]

            if answer_text is None:
                return False
        return True
    
    def _get_question_key(self, q_index: int) -> str:
        """
        Generates a key to use in the streamlit session state.

        :param q_index: Index of question in quiz.
        """
        return f"q_{q_index}"
        

class DFToDocsConverter:
    def convert_df_to_docs_with_proper_metadata(self, df: pd.DataFrame):
        """df should have the following keys:
        1. class
        2. subject
        3. page_no
        5. content
        Returns docs having content as page_content and other details in the metadata.
        """
        docs = []

        for _, row in df.iterrows():
            class_no = self._convert_df_val_to_int(row[CLASS_KEY])
            page_no = self._convert_df_val_to_int(row[PAGE_NO_KEY])
            subject = row[SUBJECT_KEY]
            content = row[CONTENT_KEY]

            doc = Document(page_content=content)
            doc.metadata = {'page': page_no, CLASS_KEY: class_no, SUBJECT_KEY: subject}
            docs.append(doc)

        return docs

    def _convert_df_val_to_int(self, value):
        if isinstance(value, str):
            if value.isnumeric():
                return int(value)
        
        if isinstance(value, float):
            if not pd.isna(value):
                return int(value)
            else:
                return ""
            
        else:
            return value


class Utils:
    def load_ollama_llm(self, model: str = 'gemma2:2b'):
        return OllamaLLM(model=model)
    
    def load_pandas_df(self, df_path: str) -> pd.DataFrame:
        if self.is_valid_csv_path(df_path):
            return pd.read_csv(df_path)
        
    def is_valid_csv_path(self, path: str):
        if is_valid_path(path):
            if path[-3:] == 'csv':
                return True
            else:
                raise Exception(f"Not a valid csv file: {path}")
        
    def get_response_from_model(llm, prompt):
        return llm.invoke(prompt)

    def get_quiz_generator_prompt():
        prompt = PromptTemplate(
            input_variables = ['no_questions', 'book_content'],
            template = QUIZ_GENERATOR_PROMPT
        )
        return prompt




UTILS = Utils()
BOOK_PATH_PROCESSOR = BookPathProcessor()
DOC_PROCESSOR = DocumentProcessor()
OCR_USER = OCRUser()
BOOKS_DETAILS_MANAGER = BookDetailsManager()
# if __name__ == "__main__":