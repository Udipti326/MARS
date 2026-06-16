# MARS: Multi-Agent Research System

MARS is an AI-powered multi-agent research platform designed to automate deep research expeditions. It coordinates multiple specialized agents (such as orchestrator, summarizer, skeptic, fact-checker, and tool executor) using LangChain and Groq to synthesize information and construct an interactive **Concept Flow Graph (CFG)**.

---

## 📂 Project Structure

This repository is structured as a monorepo consisting of two main components:

*   **`backend/`**: A Flask-based API that coordinates the multi-agent pipelines, executes search tools (Google, Wikipedia, YouTube, Arxiv), constructs the Concept Flow Graph, and saves research data into a local SQLite database.
*   **`frontend/`**: A modern React + Vite application that serves as the research dashboard, allowing users to start new research queries, view ongoing expeditions, chat contextually with research history, and visualize concept flow graphs.

---

## 🚀 Getting Started

### 1. Backend Setup (Flask API)

1.  **Navigate to the backend directory**:
    ```bash
    cd backend
    ```
2.  **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies**:
    *(Install the required packages such as Flask, LangChain, Groq integration, and database drivers)*:
    ```bash
    pip install flask flask-cors langchain-groq python-dotenv
    ```
4.  **Configure Environment Variables**:
    Create a `.env` file in the `backend/` directory:
    ```env
    GROQ_API_KEY=your-groq-api-key
    GROQ_MODEL=llama-3.1-8b-instant
    TAVILY_API_KEY=your-tavily-api-key # Optional, if using Tavily
    PORT=5001
    ```
5.  **Run the Backend Server**:
    ```bash
    python app.py
    ```
    The server will start running on `http://127.0.0.1:5001`.

---

### 2. Frontend Setup (React + Vite)

1.  **Navigate to the frontend directory**:
    ```bash
    cd ../frontend
    ```
2.  **Install Node packages**:
    ```bash
    npm install
    ```
3.  **Start the Development Server**:
    ```bash
    npm run dev
    ```
    The frontend will start running, usually at `http://localhost:5173`.

---

## 🤖 How It Works (Multi-Agent Design)

MARS runs an **ExpeditionPipeline** when you request research on a topic:
1.  **Orchestrator Agent**: Breaks down the primary query into research threads.
2.  **Tool Agent**: Interacts with web APIs (Google, Tavily, Wikipedia, Arxiv) to pull source text.
3.  **Skeptic/Fact-Checker Agent**: Evaluates the retrieved data for contradictions, hallucination, or low confidence.
4.  **Summarizer & Concept Extractor**: Pulls out key terms and defines relationships between them.
5.  **Concept Flow Graph (CFG)**: The backend synthesizes this metadata to generate an interactive graph of concepts and trends, which is rendered dynamically on the React dashboard.
