from pydantic import BaseModel, Field
from typing import Optional, Type

class QuestionSentimentAnalyzerInput(BaseModel):
    # is_ambiguous: bool = Field(..., description="If one or more of the attributes from the ambiguity criteria is missing from the question then the question is ambiguous, hence the 'is_ambiguous' parameter should return True. If ALL the attributes are identified accurately and are present in the question then is_ambiguous parameter should return false.")
    # clarification_question: str = Field(..., description="The clarification question that should be asked to the user, addressing the required attributes which are currently missing from the question. Every question should have a product name (a noun), time frame, prescription type ('nbrx', 'trx' or 'rx') and a metric to evaluate the performance (example - volume, market share etc.)")
    
    is_positive: bool = Field(
        ...,
        description="Returns a True or False value. Return True for positive or neutral queries that express curiosity, positivity, or constructive inquiry without any negative connotations. Return False for those queries with negative sentiment, unethical implications or harmful intent that may imply harm to others, unethical behavior, discrimination, or any form of malicious intent."
    )


class QuestionSentimentAnalyzerTool(BaseTool):
    """Text-to-API tool for ambiguity checking"""
    
    name: str = "question_sentiment_checker"
    args_schema: Type[BaseModel] = QuestionSentimentAnalyzerInput
    description: str = """
        This tool analyzes question sentiment, categorizing each as positive, negative, or unethical.
        It's designed to assess linguistic cues and context, providing a boolean output—True for positive or neutral sentiments and False for negative, unethical, or harmful inquiries. Ideal for filtering content or tailoring responses to ensure engagement remains constructive and ethical.
    """

    def _run(
        self,
        is_positive: bool,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        return {"QuestionSentimentAnalyzerTool": is_positive}
