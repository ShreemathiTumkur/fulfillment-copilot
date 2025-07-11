
# Fulfillment-Copilot 🚚🤖
RAG-powered seller-support agent • 14-day side project

[![Open Demo](https://img.shields.io/badge/Live_Demo-Streamlit-brightgreen?logo=streamlit)](https://fulfillment-copilot.streamlit.app)


<p align="center">
  <img src="demo.gif" width="600">
</p>


---

## Tech stack & dataset

| Layer      | Choice                                          | Why |
|------------|-------------------------------------------------|-----|
| Data       | **Smart Logistics Supply-Chain Dataset** (Kaggle) | 1 000 real delay reasons |
| Vector DB  | **FAISS**                                       | fast in-process similarity search |
| Embedding  | MiniLM-L6-v2 (sentence-transformers)            | light 384-d vectors |
| LLM        | **GPT-4o-mini**                                 | concise, cheap |
| UI         | **Streamlit**                                   | 1-file web app |
| Cloud      | AWS S3 + Glue/Athena                            | simple, pay-as-you-go |

---

## Architecture

```mermaid
graph TD
    subgraph AWS
        S3[(Parquet<br/>shipments.parquet)]
        Glue((Glue Crawler))
        Athena((Athena SQL))
        S3 --> Glue --> Athena
    end

    S3 --> E[Embeddings<br/>(MiniLM)]
    E --> F[FAISS index]

    Q[User query] --> QE[Embed query] --> F
    F --> K[Top-K passages] --> LLM[GPT-4o-mini] --> A[Answer<br/>Keyword]
    Q --> UI[Streamlit app]
    A --> UI
    K --> UI




## Evaluation

| Date       | Test set size | Accuracy |
|------------|---------------|----------|
| 2025-07-09 | 15 questions  | **53 %** |

_Current model_: gpt-4o-mini • _Retrieval_: FAISS top-5 • _Prompt_ forces Weather/Traffic/Inventory keyword.

