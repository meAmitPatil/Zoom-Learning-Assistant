# Zoom Learning Assistant 📚🤖

The Zoom Learning Assistant is an AI-powered chatbot integrated directly into the Zoom platform, designed to enhance the learning experience for students. This assistant provides lecture summaries, generates quizzes, suggests additional resources, and even saves generated notes to Google Drive for easy access—all within Zoom!

## 🚀 Project Overview

This project leverages state-of-the-art AI tools and the Zoom API to deliver an all-in-one learning experience:
- **Summarizes Lecture Content**: Quickly pull key points and highlights.
- **Generates Interactive Quizzes**: Engaging multiple-choice questions based on lecture content.
- **Additional Learning Resources**: Suggests relevant articles, videos, and books.
- **Note Saving**: Auto-saves notes to Google Drive for easy access and organization.

## 🔧 Key Technologies Used

- **Qdrant**: Used as a vector database to store and retrieve lecture segments.
- **LlamaIndex**: Structures and organizes lecture data for seamless querying and improved information retrieval.
- **Fireworks AI**: Powers quiz generation and additional resource suggestions based on lecture content.
- **Flask & Zoom API**: Handles commands directly within Zoom to enable real-time learning features.

## 🎯 Features

1. **Summarize Lecture Content**: Get a concise overview of lectures.
2. **Generate Quizzes**: Reinforce learning with multiple-choice questions.
3. **Additional Resources**: Discover supplementary materials on lecture topics.
4. **Save Notes to Google Drive**: Auto-upload generated notes to a designated folder.

## 📂 Architecture
<img width="898" alt="Screenshot 2024-10-16 at 12 51 51 PM" src="https://github.com/user-attachments/assets/3d4596ee-317c-4850-8918-ba8671321818">


## 💻 Usage

**Use the Zoom Chat Commands**:
   - `/summarize lecture` – Summarizes the lecture content.
   - `/start quiz` – Generates a quiz question from the summary.
   - `/generate notes` – Creates detailed lecture notes and saves them to Google Drive.
   - `/additional resources` – Provides extra learning resources.


## 🚀 Future Enhancements

Currently, a mock transcript is used for demonstrations. In the future, we plan to enable real-time transcript retrieval from Zoom, which will allow the assistant to process live lecture content. This feature is only available with Zoom’s paid accounts and is under development.

## 🛠 Acknowledgements

Big thanks to the contributors of Qdrant, LlamaIndex, Fireworks AI, and the Zoom Developer community for their invaluable support and tools.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Let's redefine learning with AI! 🎓💡
