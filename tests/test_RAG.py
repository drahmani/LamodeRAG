from src.rag_gpu import generate_answer

def test_generate_answer_returns_string():
    query = "I have blue eyes. What should I wear?"
    answer = generate_answer(query)
    assert isinstance(answer, str)