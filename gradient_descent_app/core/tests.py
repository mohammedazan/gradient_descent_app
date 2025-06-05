from django.test import TestCase

# Create your tests here.

class GradientDescentAppTestCase(TestCase):
    """
    Test cases for the Gradient Descent application
    """
    
    def test_home_page(self):
        """Test that the home page loads correctly"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'تطبيق Gradient Descent الرسومي')
    
    def test_upload_page(self):
        """Test that the upload page loads correctly"""
        response = self.client.get('/upload/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'رفع ملف البيانات')
    
    # Additional tests can be added here for:
    # - File upload functionality
    # - Data processing functions
    # - Gradient descent algorithms
    # - Form validation