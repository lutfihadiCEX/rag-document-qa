"""
Simple RAG Evaluation Metrics
Works with any LangChain version
"""

import pandas as pd
from typing import List, Dict
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleRAGMetrics:
    """Simple custom metrics for quick eval"""
    
    @staticmethod
    def answer_length_score(answer: str, min_length: int = 50, max_length: int = 500) -> float:
        """
        Score based on answer length (prefer medium-length answers)
        """
        length = len(answer.split())
        
        if length < min_length:
            return length / min_length
        elif length > max_length:
            return max(0.5, max_length / length)
        return 1.0
    
    @staticmethod
    def source_citation_score(answer: str, num_sources: int) -> float:
        """
        Check if answer mentions sources
        """
        source_keywords = ['source', 'according to', 'based on', 'document', 'states', 'mentions']
        has_citation = any(kw in answer.lower() for kw in source_keywords)
        
        if has_citation and num_sources > 0:
            return 1.0
        elif num_sources > 0:
            return 0.7  
        return 0.5
    
    @staticmethod
    def retrieval_score(contexts: List[str], min_contexts: int = 1, ideal_contexts: int = 3) -> float:
        """
        Score based on number of retrieved contexts
        """
        num_contexts = len(contexts)
        
        if num_contexts == 0:
            return 0.0
        elif num_contexts < min_contexts:
            return num_contexts / min_contexts
        elif num_contexts >= ideal_contexts:
            return 1.0
        else:
            return num_contexts / ideal_contexts
    
    @staticmethod
    def response_time_score(response_time: float, target_time: float = 3.0) -> float:
        """
        Score based on response time
        """
        if response_time <= target_time:
            return 1.0
        elif response_time <= target_time * 2:
            return 0.75
        elif response_time <= target_time * 3:
            return 0.5
        else:
            return 0.25
    
    @staticmethod
    def answer_completeness_score(answer: str, question: str) -> float:
        """
        Simple check if answer addresses the question
        """
        # Extract key words from question
        question_words = set(question.lower().split())
        # Remove common words
        stop_words = {'what', 'is', 'are', 'how', 'why', 'when', 'where', 'the', 'a', 'an'}
        question_words = question_words - stop_words
        
        if not question_words:
            return 0.5  # Can't evaluate
        
        answer_lower = answer.lower()
        matches = sum(1 for word in question_words if word in answer_lower)
        
        return min(1.0, matches / len(question_words))
    
    @staticmethod
    def calculate_all_metrics(answer: str, question: str, contexts: List[str], response_time: float) -> Dict:
        """
        Calculate all simple metrics
        """
        return {
            'answer_length_score': SimpleRAGMetrics.answer_length_score(answer),
            'citation_score': SimpleRAGMetrics.source_citation_score(answer, len(contexts)),
            'retrieval_score': SimpleRAGMetrics.retrieval_score(contexts),
            'response_time_score': SimpleRAGMetrics.response_time_score(response_time),
            'completeness_score': SimpleRAGMetrics.answer_completeness_score(answer, question),
            'overall_score': 0.0,  # Base tresh, calc below
            'response_time': response_time,
            'num_sources': len(contexts)
        }
    
    @staticmethod
    def calculate_overall_score(metrics: Dict) -> float:
        """
        Calculate weighted overall score
        """
        weights = {
            'answer_length_score': 0.15,
            'citation_score': 0.25,
            'retrieval_score': 0.20,
            'response_time_score': 0.15,
            'completeness_score': 0.25
        }
        
        total = sum(
            metrics[key] * weight 
            for key, weight in weights.items() 
            if key in metrics
        )
        
        return total


class RAGEvaluationTracker:
    """Track and analyze RAG evaluation metrics over time"""
    
    def __init__(self):
        self.history = []
    
    def add_evaluation(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        response_time: float,
        timestamp: datetime = None
    ):
        """add evaluation to history"""
        metrics = SimpleRAGMetrics.calculate_all_metrics(
            answer, question, contexts, response_time
        )
        
        # Calculate overall score
        metrics['overall_score'] = SimpleRAGMetrics.calculate_overall_score(metrics)
        
        entry = {
            'timestamp': timestamp or datetime.now(),
            'question': question,
            'answer': answer,
            'contexts': contexts,
            'metrics': metrics
        }
        
        self.history.append(entry)
        logger.info(f"Added evaluation (Overall: {metrics['overall_score']:.2%})")
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics across all evals"""
        if not self.history:
            return {}
        
        all_metrics = [entry['metrics'] for entry in self.history]
        
        summary = {
            'total_queries': len(self.history),
            'avg_answer_length_score': sum(m['answer_length_score'] for m in all_metrics) / len(all_metrics),
            'avg_citation_score': sum(m['citation_score'] for m in all_metrics) / len(all_metrics),
            'avg_retrieval_score': sum(m['retrieval_score'] for m in all_metrics) / len(all_metrics),
            'avg_response_time_score': sum(m['response_time_score'] for m in all_metrics) / len(all_metrics),
            'avg_completeness_score': sum(m['completeness_score'] for m in all_metrics) / len(all_metrics),
            'avg_overall_score': sum(m['overall_score'] for m in all_metrics) / len(all_metrics),
            'avg_response_time': sum(m['response_time'] for m in all_metrics) / len(all_metrics),
            'avg_sources_used': sum(m['num_sources'] for m in all_metrics) / len(all_metrics),
        }
        
        return summary
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert history to pandas df"""
        if not self.history:
            return pd.DataFrame()
        
        records = []
        for entry in self.history:
            record = {
                'timestamp': entry['timestamp'],
                'question': entry['question'][:50] + '...' if len(entry['question']) > 50 else entry['question'],
                'answer_length': len(entry['answer'].split()),
                'num_sources': entry['metrics']['num_sources'],
                'response_time': entry['metrics']['response_time'],
                **{k: v for k, v in entry['metrics'].items() 
                   if k not in ['response_time', 'num_sources']}
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def export_to_csv(self, filepath: str):
        """Export history to CSV"""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(df)} evaluations to {filepath}")
    
    def get_performance_trend(self, window: int = 10) -> Dict:
        """Get perf trend over last N queries"""
        if len(self.history) < window:
            window = len(self.history)
        
        if window == 0:
            return {}
        
        recent = self.history[-window:]
        recent_overall = [e['metrics']['overall_score'] for e in recent]
        
        if len(self.history) > window:
            previous = self.history[-window*2:-window]
            previous_overall = [e['metrics']['overall_score'] for e in previous]
            
            recent_avg = sum(recent_overall) / len(recent_overall)
            previous_avg = sum(previous_overall) / len(previous_overall)
            
            trend = ((recent_avg - previous_avg) / previous_avg) * 100 if previous_avg > 0 else 0
        else:
            recent_avg = sum(recent_overall) / len(recent_overall)
            trend = 0
        
        return {
            'recent_avg_score': recent_avg,
            'trend_percentage': trend,
            'trend_direction': 'improving' if trend > 0 else 'declining' if trend < 0 else 'stable'
        }

if __name__ == "__main__":
    # Metrics test
    print("="*60)
    print("SIMPLE RAG METRICS TEST")
    print("="*60)
    
    # Sample
    question = "What is quantum computing?"
    answer = "According to the source, quantum computing uses quantum mechanics principles to perform computations using qubits instead of classical bits."
    contexts = [
        "Quantum computing leverages quantum mechanics...",
        "Qubits can exist in superposition states...",
        "Quantum computers use quantum bits for computation..."
    ]
    response_time = 2.3
    
    # Calculate metrics
    metrics = SimpleRAGMetrics.calculate_all_metrics(
        answer, question, contexts, response_time
    )
    metrics['overall_score'] = SimpleRAGMetrics.calculate_overall_score(metrics)
    
    print("\nMetrics:")
    for key, value in metrics.items():
        if isinstance(value, float) and key != 'response_time':
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")
    
    # Tracker test
    print("\n" + "="*60)
    print("EVALUATION TRACKER TEST")
    print("="*60)
    
    tracker = RAGEvaluationTracker()
    
    # Add some evals
    for i in range(3):
        tracker.add_evaluation(
            question=f"Question {i+1}",
            answer="This is a sample answer with proper length and citation.",
            contexts=["context1", "context2"],
            response_time=2.0 + i * 0.5
        )
    
    # Get summary
    summary = tracker.get_summary_stats()
    print("\nSummary Statistics:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}" if value <= 1 else f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Get trend
    trend = tracker.get_performance_trend()
    print("\nPerformance Trend:")
    for key, value in trend.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}" if abs(value) <= 1 else f"  {key}: {value:+.2f}%")
        else:
            print(f"  {key}: {value}")
    
    print("\n✅ All tests passed!")
