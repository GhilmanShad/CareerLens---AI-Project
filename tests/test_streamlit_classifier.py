"""Local integration smoke test; requires full original project and Streamlit."""
import importlib.util
import unittest
from pathlib import Path

@unittest.skipUnless(importlib.util.find_spec('streamlit') is not None, 'Streamlit is not installed')
class StreamlitClassifierTests(unittest.TestCase):
    def test_predict_without_loading_dataset(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30).run()
        self.assertFalse(app.exception)
        app.sidebar.radio[0].set_value('Experience Classifier (SVM)').run()
        self.assertFalse(app.exception)
        app.text_input[0].set_value('Software Engineer Intern')
        app.text_area[0].set_value('Summer internship for students')
        next(b for b in app.button if b.label == 'Predict experience level').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('Predicted job level:' in s.value for s in app.success))
