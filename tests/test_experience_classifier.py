import unittest
from modules.experience_classifier import load_model, predict_level

class ExperienceClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()
    def test_all_six_classes(self):
        self.assertEqual(set(self.model['classifier'].classes_), {'Associate','Director','Entry level','Executive','Internship','Mid-Senior level'})
    def test_inference(self):
        for title,description,company in [('Software Engineer Intern','Summer internship for students','Example'),('Senior Engineer','5 years of professional experience','Unseen company'),('Sales Associate','','')]:
            with self.subTest(title=title):
                self.assertIn(predict_level(self.model,title,description,company), self.model['classifier'].classes_)
    def test_missing_title(self):
        with self.assertRaises(ValueError):
            predict_level(self.model, '   ')
    def test_determinism(self):
        args=('Software Developer','3 years of experience','Example')
        self.assertEqual(predict_level(self.model,*args),predict_level(self.model,*args))
