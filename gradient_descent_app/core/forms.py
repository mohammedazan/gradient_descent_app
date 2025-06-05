from django import forms
from django.template.defaulttags import register

class UploadFileForm(forms.Form):
    file = forms.FileField(
        label="اختر ملف CSV",
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.csv'
        })
    )

class PreprocessingForm(forms.Form):
    MISSING_VALUE_CHOICES = [
        ('drop', 'حذف الصفوف التي تحتوي على قيم مفقودة'),
        ('mean', 'ملء بالمتوسط'),
        ('constant', 'ملء بقيمة ثابتة'),
    ]
    
    SCALING_CHOICES = [
        ('standard', 'التطبيع المعياري (Standard Scaling)'),
        ('minmax', 'تطبيع Min-Max'),
    ]
    
    missing_values = forms.ChoiceField(
        choices=MISSING_VALUE_CHOICES,
        label="معالجة القيم المفقودة",
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    fill_value = forms.FloatField(
        required=False,
        label="القيمة الثابتة للملء",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'أدخل القيمة الثابتة'
        })
    )
    
    scaling_method = forms.ChoiceField(
        choices=SCALING_CHOICES,
        label="طريقة التطبيع",
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    categorical_encoding = forms.BooleanField(
        required=False,
        label="تشفير المتغيرات الفئوية",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

class AlgorithmForm(forms.Form):
    MODEL_CHOICES = [
        ('linear', 'الانحدار الخطي (Linear Regression)'),
        ('logistic', 'الانحدار اللوجستي (Logistic Regression)'),
    ]
    
    model_type = forms.ChoiceField(
        choices=MODEL_CHOICES,
        label="نوع النموذج",
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    learning_rate = forms.FloatField(
        initial=0.01,
        label="معدل التعلم (α)",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.001',
            'min': '0.001',
            'max': '1.0'
        })
    )
    
    num_iterations = forms.IntegerField(
        initial=1000,
        label="عدد التكرارات",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '100',
            'max': '10000'
        })
    )