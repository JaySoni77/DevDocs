import dspy

class QueryRewriter(dspy.Signature):
    """
    Rewrite a vague user query into a precise, keyword-rich search query 
    optimized for technical documentation retrieval.
    """
    question = dspy.InputField()
    optimized_query = dspy.OutputField(desc="A precise, keyword-heavy retrieval query.")

class DocGrader(dspy.Signature):
    """
    Evaluate if the provided technical context is sufficient to answer the question.
    """
    question = dspy.InputField()
    context = dspy.InputField()
    is_sufficient = dspy.OutputField(desc="YES or NO")

class AnswerGenerator(dspy.Signature):
    """
    Generate a structured, technical answer based ONLY on the provided context with citations.
    """
    question = dspy.InputField()
    context = dspy.InputField()
    answer = dspy.OutputField(desc="A grounded, cited technical answer in Markdown.")
