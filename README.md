# AI Legal Assistant 🤖⚖️

A sophisticated legal assistant powered by artificial intelligence that helps users understand legal matters, court judgments, and legal procedures in an interactive chat interface.

## Demo 🎥

<div align="center">
  <a href="https://drive.google.com/file/d/1qF4q1h9cEBi3WywsN15LzNV97mt1yfek/view?usp=sharing">
    <img src="https://drive.google.com/uc?export=view&id=1qF4q1h9cEBi3WywsN15LzNV97mt1yfek" alt="AI Legal Assistant Demo" width="600"/>
  </a>
</div>

<p align="center">
  <i>👆 Click to watch the full demo video</i>
</p>

## Key Features 🌟

- **Interactive Chat Interface**: WhatsApp-style chat interface for natural conversations
- **Multiple Persona Modes**:
  - Layperson (simple, non-technical)
  - Legal Professional (detailed, citations)
  - Concise (short answers)
  - Verbose (step-by-step explanations)
- **Comprehensive Legal Information**:
  - Case summaries and query overviews
  - Step-by-step actionable items
  - Applicable laws and regulations
  - Legal procedures and documentation
  - Professional guidance
  
## Technologies Used 🛠️

### Core Technologies
- **Python** with **Streamlit** for the web interface
- **Groq** for advanced language model capabilities
- **FAISS** for efficient vector similarity search
- **HuggingFace** for embeddings (sentence-transformers)
- **LangChain** for AI/LLM application framework

### Key Libraries and Tools
- **langchain_groq**: For integration with Groq's language models
- **langchain_community**: For document loading and processing
- **PyPDF**: For PDF document processing
- **Wikipedia API**: For accessing legal references
- **Arxiv API**: For academic legal research
- **CUDA Support**: GPU acceleration when available

## System Architecture 🏗️

1. **Vector Store System**:
   - Efficient document indexing
   - Persistent storage with pickle
   - MMR (Maximal Marginal Relevance) search
   
2. **Document Processing**:
   - PDF directory loading
   - Web content integration
   - Recursive text splitting
   
3. **Chat System**:
   - History management
   - Multiple persona support
   - Real-time response generation

## Getting Started 🚀

1. Clone the repository
```bash
git clone https://github.com/yourusername/LegelAI.git
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables
```bash
GROQ_API_KEY=your_api_key_here
```

4. Run the application
```bash
streamlit run app.py
```

## Use Cases 💡

1. **Legal Research**:
   - Quick access to relevant laws and regulations
   - Finding applicable court judgments
   - Understanding legal procedures

2. **Legal Documentation**:
   - Step-by-step guidance for filing requirements
   - Document preparation assistance
   - Timeline expectations

3. **Professional Guidance**:
   - Recommendations for legal professionals
   - Legal aid options
   - Risk assessment

## Project Structure 📁

```
LegelAI/
├── app.py              # Main application file
├── legal.py           # Legal processing modules
├── data/              # PDF and legal document storage
├── vector_store.pkl   # Persistent vector storage
└── README.md          # Project documentation
```

## Future Enhancements 🔮

- Integration with more legal databases
- Multi-language support
- Document generation capabilities
- Advanced case law analysis
- Mobile application development

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📝

This project is licensed under the MIT License - see the LICENSE file for details.

---

⭐ Don't forget to star this repository if you found it helpful!