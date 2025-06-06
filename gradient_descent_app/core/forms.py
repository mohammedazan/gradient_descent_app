from django import forms
from django.template.defaulttags import register

class UploadFileForm(forms.Form):
    file = forms.FileField(
        label="Choisissez un fichier CSV",
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.csv'
        })
    )

class PreprocessingForm(forms.Form):
    MISSING_VALUE_CHOICES = [
        ('drop', 'Supprimer les lignes contenant des valeurs manquantes'),
        ('mean', 'Remplir avec la moyenne'),
        ('constant', 'Remplir avec une valeur constante'),
    ]
    
    SCALING_CHOICES = [
        ('standard', 'Normalisation standard (Standard Scaling)'),
        ('minmax', 'Normalisation Min-Max'),
    ]
    
    missing_values = forms.ChoiceField(
        choices=MISSING_VALUE_CHOICES,
        label="Traitement des valeurs manquantes",
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    fill_value = forms.FloatField(
        required=False,
        label="Valeur constante de remplissage",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Entrez la valeur constante'
        })
    )
    
    scaling_method = forms.ChoiceField(
        choices=SCALING_CHOICES,
        label="Méthode de normalisation",
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    categorical_encoding = forms.BooleanField(
        required=False,
        label="Encodage des variables catégorielles",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

class AlgorithmForm(forms.Form):
    MODEL_CHOICES = [
        ('linear', 'Régression linéaire (Linear Regression)'),
        ('logistic', 'Régression logistique (Logistic Regression)'),
    ]
    
    model_type = forms.ChoiceField(
        choices=MODEL_CHOICES,
        label="Type de modèle",
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    learning_rate = forms.FloatField(
        initial=0.01,
        label="Taux d'apprentissage (α)",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.001',
            'min': '0.001',
            'max': '1.0'
        })
    )
    
    num_iterations = forms.IntegerField(
        initial=1000,
        label="Nombre d'itérations",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '100',
            'max': '10000'
        })
    )