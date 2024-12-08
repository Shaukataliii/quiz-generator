### README for Quiz Generator Project  

# **AI-Powered Quiz Generator**  
Effortlessly generate quizzes, evaluate student performance, and enhance learning experiences using advanced AI techniques like Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs).  

---

## **Project Overview**  
This project leverages cutting-edge AI technologies to streamline quiz creation and simplify the evaluation process. By integrating tools like LangChain and Chroma for vector-based data retrieval and GPT models for quiz generation, the system provides educators with a robust solution to modernize traditional teaching methods.  

---

## **Key Features**  
- **Quiz Customization**:  
  - Select grade, book, chapter, difficulty level, and number of questions.  
- **AI-Powered Generation**:  
  - Combines RAG and GPT models for contextually accurate quiz creation.  
- **Scalable Architecture**:  
  - Easily adaptable for various subjects and grades.  

---

## **How It Works**  
1. **Book Storage**:  
   - Books for different grades and subjects are stored in a vector database using Chroma.  
2. **Quiz Generation**:  
   - Based on user inputs (grade, book, chapter, difficulty, and number of questions), the system retrieves relevant content and generates questions using GPT models.  
3. **Results**:  
   - Displays quiz outcomes to evaluate performance.  

---

## **Future Enhancements**  
- **Performance Tracking**: Aggregate data over time for individual and group progress analysis.  
- **Adaptive Learning**: Provide personalized feedback and recommendations based on quiz results.  
- **Enhanced User Interface**: Add more intuitive features for teachers and students.  

---

## **Getting Started**  

### **Requirements**  
- **Python** 3.8 or higher  
- **Libraries**: Install dependencies using the provided `requirements.txt` file  

```bash
pip install -r requirements.txt
```

### **Setup**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/quiz-generator.git
   cd quiz-generator
   ```
2. Update your Google API in the .env file.
3. Run the application:  
   ```bash
   streamlit run 0-app.py
   ```

---

## **Usage**  
1. Select the grade and subject.  
2. Specify the chapter, difficulty level, and number of questions.  
3. Generate the quiz and view the results.  

---

## **Contributing**  
Contributions are welcome! If you’d like to improve this project, please fork the repository and submit a pull request.  

---

## **License**  
This project is licensed under the MIT License. See the `LICENSE` file for details.  

---

## **Related Blog Post**  
For an in-depth overview of this project, check out the [blog post](https://shaukat.tech/quiz-generator-revolutionizing-education-with-ai/).  
