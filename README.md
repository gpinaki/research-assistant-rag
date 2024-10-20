```markdown
# Research Assistant ChatBot 

A powerful, intelligent research assistant built with LangChain and Streamlit that 
uses RAG (Retrieval Augmented Generation) 
to provide context-aware responses based on your documents.

## 🌟 Features

- **Multi-Format Document Processing**: Support for PDF, Excel, and CSV files
- **Intelligent Response Generation**: Powered by GPT-4 and RAG architecture
- **Interactive UI**: Clean, user-friendly interface built with Streamlit
- **Real-time Response Streaming**: See AI responses as they're generated
- **Source Attribution**: Transparent reference to source documents
- **Theme Customization**: Light and dark mode support
- **Efficient Document Processing**: Optimized chunk processing and retrieval

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/research-assistant-chatbot.git
cd research-assistant-chatbot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create an `api_credentials.yml` file in the root directory:
```yaml
openai_api_key: "your-api-key-here"
```

## 🚀 Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Upload your research documents (PDF, Excel, or CSV)

4. Start asking questions about your documents!

## 💡 Example Questions

- "What are the key findings in the uploaded research papers?"
- "Can you summarize the data from the Excel sheets?"
- "What are the main trends in the CSV data?"
- "Compare the methodologies used in different papers."

## 🔧 Configuration

The application can be configured through the following environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `STREAMLIT_THEME`: Light or Dark theme preference
- `CHUNK_SIZE`: Size of document chunks for processing (default: 1500)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

Pinaki Guha  - [pinaki.guha@gmail.com](mailto:pinaki.guha@gmail.com)

Project Link: [https://github.com/gpinaki/research-assistant-rag/](https://github.com/gpinaki/research-assistant-rag)
```

