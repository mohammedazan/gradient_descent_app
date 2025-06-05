from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from .forms import UploadFileForm, PreprocessingForm, AlgorithmForm
from .utils import (
    read_dataset, get_descriptive_stats, handle_missing_values,
    scale_features, encode_categorical, split_data, add_bias_column,
    gradient_descent_linear, gradient_descent_logistic, plot_learning_curve,
    validate_csv_file, ensure_media_dirs
)
import os
import pandas as pd
import numpy as np

def home(request):
    """Home page view"""
    return render(request, 'core/home.html')

def upload_data(request):
    """Upload CSV file view"""
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            
            # Validate file
            is_valid, message = validate_csv_file(uploaded_file)
            if not is_valid:
                messages.error(request, message)
                return render(request, 'core/upload.html', {'form': form})
            
            # Ensure media directories exist
            uploads_dir, _ = ensure_media_dirs()
            
            # Save file
            file_path = os.path.join(uploads_dir, uploaded_file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            
            # Store file path in session
            request.session['csv_file_path'] = file_path
            request.session['csv_file_name'] = uploaded_file.name
            
            messages.success(request, 'Fichier téléchargé avec succès!')
            return redirect('core:dashboard')
    else:
        form = UploadFileForm()
    
    return render(request, 'core/upload.html', {'form': form})

def dashboard(request):
    """Data statistics dashboard view"""
    csv_file_path = request.session.get('csv_file_path')
    if not csv_file_path or not os.path.exists(csv_file_path):
        messages.error(request, 'Veuillez télécharger un fichier CSV d\'abord')
        return redirect('core:upload')
    
    try:
        # Read dataset
        df = read_dataset(csv_file_path)
        
        # Get descriptive statistics
        stats = get_descriptive_stats(df)
        
        # Get basic info about dataset
        dataset_info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'total_missing': df.isnull().sum().sum()
        }
        
        context = {
            'stats': stats,
            'dataset_info': dataset_info,
            'file_name': request.session.get('csv_file_name', 'Unknown')
        }
        
        return render(request, 'core/dashboard.html', context)
        
    except Exception as e:
        messages.error(request, f'Erreur de lecture du fichier : {str(e)}')
        return redirect('core:upload')

def preprocessing(request):
    """Data preprocessing view"""
    csv_file_path = request.session.get('csv_file_path')
    if not csv_file_path or not os.path.exists(csv_file_path):
        messages.error(request, 'Veuillez télécharger un fichier CSV d\'abord')
        return redirect('core:upload')
    
    if request.method == 'POST':
        form = PreprocessingForm(request.POST)
        if form.is_valid():
            try:
                # Read original dataset
                df = read_dataset(csv_file_path)
                
                # Handle missing values
                missing_method = form.cleaned_data['missing_values']
                fill_value = form.cleaned_data.get('fill_value')
                df = handle_missing_values(df, missing_method, fill_value)
                
                # Scale features
                scaling_method = form.cleaned_data['scaling_method']
                df = scale_features(df, scaling_method)
                
                # Encode categorical variables
                if form.cleaned_data['categorical_encoding']:
                    df = encode_categorical(df)
                
                # Save processed dataset
                _, processed_dir = ensure_media_dirs()
                processed_file_path = os.path.join(processed_dir, 'processed.csv')
                df.to_csv(processed_file_path, index=False)
                
                # Store processed file path in session
                request.session['processed_file_path'] = processed_file_path
                
                messages.success(request, 'Préparation exécutée avec succès!')
                return redirect('core:algorithm')
                
            except Exception as e:
                messages.error(request, f'Erreur de traitement des données : {str(e)}')
    else:
        form = PreprocessingForm()
    
    return render(request, 'core/preprocessing.html', {'form': form})

def algorithm(request):
    """Run gradient descent algorithm view"""
    processed_file_path = request.session.get('processed_file_path')
    if not processed_file_path or not os.path.exists(processed_file_path):
        messages.error(request, 'Veuillez exécuter la préparation d\'abord')
        return redirect('core:preprocessing')
    
    if request.method == 'POST':
        form = AlgorithmForm(request.POST)
        if form.is_valid():
            try:
                # Read processed dataset
                df = read_dataset(processed_file_path)
                
                # Split data (use last column as target)
                X_train, X_test, y_train, y_test = split_data(df, df.columns[-1])
                
                # Convert to numpy arrays
                X_train = X_train.values
                X_test = X_test.values
                y_train = y_train.values
                y_test = y_test.values
                
                # Add bias column
                X_train = add_bias_column(X_train)
                X_test = add_bias_column(X_test)
                
                # Get form parameters
                model_type = form.cleaned_data['model_type']
                learning_rate = form.cleaned_data['learning_rate']
                num_iterations = form.cleaned_data['num_iterations']
                
                # Run gradient descent
                if model_type == 'linear':
                    theta, cost_history = gradient_descent_linear(
                        X_train, y_train, learning_rate, num_iterations
                    )
                    
                    # Make predictions
                    predictions = X_test.dot(theta)
                    
                    # Calculate MSE
                    mse = np.mean((predictions - y_test) ** 2)
                    performance = {'metric': 'MSE', 'value': mse}
                    
                else:  # logistic
                    theta, cost_history = gradient_descent_logistic(
                        X_train, y_train, learning_rate, num_iterations
                    )
                    
                    # Make predictions
                    from .utils import sigmoid
                    probabilities = sigmoid(X_test.dot(theta))
                    predictions = (probabilities >= 0.5).astype(int)
                    
                    # Calculate accuracy
                    accuracy = np.mean(predictions == y_test)
                    performance = {'metric': 'Accuracy', 'value': accuracy}
                
                # Generate learning curve plot
                learning_curve_image = plot_learning_curve(cost_history)
                
                # Store results in session
                request.session['results'] = {
                    'theta': theta.tolist(),
                    'performance': performance,
                    'learning_curve_image': learning_curve_image,
                    'model_type': model_type,
                    'learning_rate': learning_rate,
                    'num_iterations': num_iterations
                }
                
                return redirect('core:results')
                
            except Exception as e:
                messages.error(request, f'Erreur d\'exécution de l\'algorithme : {str(e)}')
    else:
        form = AlgorithmForm()
    
    return render(request, 'core/algorithm.html', {'form': form})

def results(request):
    """Display algorithm results view"""
    results = request.session.get('results')
    if not results:
        messages.error(request, 'Aucun résultat à afficher')
        return redirect('core:algorithm')
    
    return render(request, 'core/results.html', {'results': results})