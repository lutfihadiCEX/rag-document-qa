"""
RAG Document QA - Streamlit UI with Evaluation
"""

import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime
import tempfile

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.ingestion import DocumentIngestion
from app.pipeline import RAGPipeline
from app.evaluation import SimpleRAGMetrics, RAGEvaluationTracker

st.set_page_config(
    page_title="RAG Document QA",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Document Q&A System")
st.markdown("*Upload documents and ask questions - with evaluation!*")

with st.sidebar:
    st.header("⚙️ Configuration")
    model = st.selectbox("Select Model", ["llama3.2", "mistral", "phi3"])
    st.header("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'evaluation_tracker' not in st.session_state:
    st.session_state.evaluation_tracker = RAGEvaluationTracker()


if uploaded_files and st.button("🔄 Process Documents"):
    with st.spinner("Processing..."):
        try:
            temp_dir = tempfile.mkdtemp()
            file_paths = []
            
            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                file_paths.append(file_path)
            
            ingestor = DocumentIngestion()
            vectorstore, num_chunks = ingestor.process_documents(file_paths)
            
            st.session_state.vectorstore = vectorstore
            st.session_state.rag_pipeline = RAGPipeline(vectorstore, model_name=model)
            
            st.success(f"✅ Processed {len(file_paths)} documents into {num_chunks} chunks!")
        except Exception as e:
            st.error(f"Error: {str(e)}")


if st.session_state.rag_pipeline:
    tabs = st.tabs(["💬 Ask Questions", "📊 Evaluation", "📜 History"])
    

    with tabs[0]:
        st.header("💬 Ask Questions")
        
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is the main topic?",
            key="question_input"
        )
        
        ask_button = st.button("🚀 Ask", type="primary")
        
        if ask_button and question:
            with st.spinner("🤔 Thinking..."):
                try:
                    # Query RAG
                    result = st.session_state.rag_pipeline.query(question)
                    
                    # Extract contexts
                    contexts = [doc.page_content for doc in result["sources"]]
                    
                    # Add to eval tracker
                    st.session_state.evaluation_tracker.add_evaluation(
                        question=question,
                        answer=result["answer"],
                        contexts=contexts,
                        response_time=result["response_time"]
                    )
                    
                    # Add to chat hist
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": result["answer"],
                        "sources": result["sources"],
                        "response_time": result["response_time"]
                    })
                    
                    st.success("✅ Answer generated!")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        if st.session_state.chat_history:
            st.divider()
            st.subheader("Recent Answers")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
                with st.expander(f"Q: {chat['question'][:60]}...", expanded=(i==1)):
                    st.markdown(f"**Answer:** {chat['answer']}")
                    st.caption(f"⏱️ {chat['response_time']:.2f}s")
                    
                with st.expander("📄 Sources"):
                    for j, source in enumerate(chat['sources'][:3], 1):
                        st.text(f"Source {j}: {source.page_content[:200]}...")
    

    with tabs[1]:
        st.header("📊 RAG System Evaluation")
        
        tracker = st.session_state.evaluation_tracker
        
        if len(tracker.history) == 0:
            st.info("👆 Ask questions in the first tab to see evaluation metrics!")
        else:
            # Summary stats
            summary = tracker.get_summary_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Overall Score", f"{summary['avg_overall_score']:.1%}")
            with col2:
                st.metric("Completeness", f"{summary['avg_completeness_score']:.1%}")
            with col3:
                st.metric("Citation Rate", f"{summary['avg_citation_score']:.1%}")
            with col4:
                st.metric("Avg Response", f"{summary['avg_response_time']:.2f}s")
            
            st.divider()
            
            # Met details
            st.subheader("📋 Query History")
            
            df = tracker.to_dataframe()
            
            display_df = df[[
                'timestamp',
                'question',
                'overall_score',
                'completeness_score',
                'citation_score',
                'response_time'
            ]].copy()
            
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
            display_df['overall_score'] = display_df['overall_score'].apply(lambda x: f"{x:.1%}")
            display_df['completeness_score'] = display_df['completeness_score'].apply(lambda x: f"{x:.1%}")
            display_df['citation_score'] = display_df['citation_score'].apply(lambda x: f"{x:.1%}")
            display_df['response_time'] = display_df['response_time'].apply(lambda x: f"{x:.2f}s")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.divider()
            
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Evaluation Report",
                csv,
                f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
    

    with tabs[2]:
        st.header("📜 Full Conversation History")
        
        if st.session_state.chat_history:
            for i, chat in enumerate(st.session_state.chat_history, 1):
                st.markdown(f"### Question {i}")
                st.markdown(f"**Q:** {chat['question']}")
                st.markdown(f"**A:** {chat['answer']}")
                st.caption(f"Response time: {chat['response_time']:.2f}s")
                st.divider()
        else:
            st.info("No conversation history yet")

else:
    # No docs processed
    st.info("👈 Upload documents in the sidebar to get started!")

st.divider()
st.caption("🚀 RAG Document QA with Evaluation | Powered by LangChain + Ollama")
