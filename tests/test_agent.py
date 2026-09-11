import unittest
import pandas as pd
from src.classifier.intent_classifier import clean_text, assign_intent_rules
from src.escalation.decision import decide_escalation


class TestSpotifySupportAgent(unittest.TestCase):
    def test_clean_text(self):
        raw = "Hey @SpotifyCares I can't login https://spotify.com/help   please help!"
        cleaned = clean_text(raw)
        self.assertNotIn("@SpotifyCares", cleaned)
        self.assertNotIn("https://spotify.com/help", cleaned)
        self.assertIn("login", cleaned)

    def test_intent_rules_login(self):
        query = "I forgot my password and cannot log in to my account"
        intent = assign_intent_rules(query)
        self.assertEqual(intent, "Account / Login")

    def test_intent_rules_billing(self):
        query = "I was charged twice for Spotify Premium subscription"
        intent = assign_intent_rules(query)
        self.assertEqual(intent, "Premium / Billing / Subscription")

    def test_intent_rules_streaming(self):
        query = "Songs keep pausing and buffering every few seconds"
        intent = assign_intent_rules(query)
        self.assertEqual(intent, "Playback / Streaming Issue")

    def test_escalation_security_issue(self):
        query = "My account was hacked and password was changed"
        cases = pd.DataFrame([{"customer_query": "hacked", "support_response": "We will help"}])
        decision, reason = decide_escalation(query, "Account / Login", cases)
        self.assertEqual(decision, "HUMAN_ESCALATION")

    def test_escalation_empty_cases(self):
        query = "Random issue with something unknown"
        empty_cases = pd.DataFrame()
        decision, reason = decide_escalation(query, "General / Other Support", empty_cases)
        self.assertEqual(decision, "HUMAN_ESCALATION")


if __name__ == "__main__":
    unittest.main()
