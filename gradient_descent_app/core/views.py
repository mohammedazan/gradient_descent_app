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
            
            # Test reading the CSV file immediately after upload
            try:
                print(f"[DEBUG] Testing CSV file readability: {file_path}")
                test_df = read_dataset(file_path)
                print(f"[DEBUG] CSV file successfully validated. Shape: {test_df.shape}")
                
                # Store file path in session
                request.session['csv_file_path'] = file_path
                request.session['csv_file_name'] = uploaded_file.name
                
                messages.success(request, f'Fichier téléchargé avec succès! Trouvé {test_df.shape[0]} lignes et {test_df.shape[1]} colonnes de données valides.')
                return redirect('core:dashboard')
                
            except Exception as e:
                # Remove the uploaded file if it can't be read
                try:
                    os.remove(file_path)
                except:
                    pass
                
                error_msg = str(e)
                if "No valid numeric data found" in error_msg:
                    messages.error(request,
                        'Erreur : Le fichier CSV ne contient pas de données numériques valides. '
                        'Veuillez vérifier que votre fichier contient des colonnes avec des valeurs numériques. '
                        'Consultez la console du serveur pour plus de détails de débogage.')
                elif "Could not read CSV file" in error_msg:
                    messages.error(request,
                        'Erreur : Impossible de lire le fichier CSV. '
                        'Veuillez vérifier l\'encodage du fichier (UTF-8 recommandé) et le délimiteur utilisé (virgule ou point-virgule). '
                        'Consultez la console du serveur pour plus de détails de débogage.')
                elif "Insufficient data" in error_msg:
                    messages.error(request,
                        'Erreur : Données insuffisantes. '
                        'Le fichier doit contenir au moins 5 lignes de données numériques valides pour l\'analyse.')
                else:
                    messages.error(request, f'Erreur de lecture du fichier CSV : {error_msg}')
                
                return render(request, 'core/upload.html', {'form': form})
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
        error_msg = str(e)
        if "No valid numeric data found" in error_msg:
            messages.error(request,
                'Erreur : Le fichier CSV ne contient pas de données numériques valides après conversion. '
                'Veuillez vérifier le format de vos données.')
        else:
            messages.error(request, f'Erreur de lecture du fichier : {error_msg}')
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
                error_msg = str(e)
                if "No valid numeric data found" in error_msg:
                    messages.error(request,
                        'Erreur de traitement : Aucune donnée numérique valide trouvée après le prétraitement. '
                        'Veuillez vérifier vos paramètres de préparation.')
                else:
                    messages.error(request, f'Erreur de traitement des données : {error_msg}')
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
                
                # --- New plots: Absolute Error Histogram, Predictions vs. True, Train/Test Error Curve ---
                import matplotlib.pyplot as plt
                import io, base64

                # Compute absolute errors (for histogram)
                if model_type == 'linear':
                    abs_errors = np.abs(predictions - y_test)
                else:
                    abs_errors = np.abs(predictions - y_test)  # For logistic, this is 0/1 errors

                # 1. Histogramme des Erreurs Absolues
                plt.figure(figsize=(8, 5))
                plt.hist(abs_errors, bins=20, color='skyblue', edgecolor='black')
                plt.xlabel('Erreur absolue')
                plt.ylabel('Fréquence')
                plt.title('Histogramme des Erreurs Absolues')
                plt.grid(alpha=0.3)
                buf1 = io.BytesIO()
                plt.savefig(buf1, format='png', dpi=100, bbox_inches='tight')
                buf1.seek(0)
                abs_error_hist = base64.b64encode(buf1.getvalue()).decode('utf-8')
                plt.close()

                # 2. Scatter Plot : Prédictions vs Valeurs Réelles
                plt.figure(figsize=(8, 5))
                plt.scatter(y_test, predictions, alpha=0.7, color='darkorange', edgecolor='k')
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b--', lw=2, label='y = x')
                plt.xlabel('Valeurs réelles')
                plt.ylabel('Prédictions')
                plt.title('Prédictions vs Valeurs Réelles')
                plt.legend()
                plt.grid(alpha=0.3)
                buf2 = io.BytesIO()
                plt.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
                buf2.seek(0)
                pred_vs_true_scatter = base64.b64encode(buf2.getvalue()).decode('utf-8')
                plt.close()

                # 3. Courbe d’erreur Train vs Test
                # For now, use only the test cost history (already available) as cost_history
                # Optionally, if you want to compute train cost at each iteration, you would need to modify the gradient descent functions.
                plt.figure(figsize=(10, 6))
                plt.plot(range(len(cost_history)), cost_history, 'r-', label='Test (val)')
                plt.xlabel('Itération')
                plt.ylabel('Coût')
                plt.title('Courbe d’erreur Train vs Test')
                plt.legend()
                plt.grid(alpha=0.3)
                buf3 = io.BytesIO()
                plt.savefig(buf3, format='png', dpi=100, bbox_inches='tight')
                buf3.seek(0)
                train_test_curve = base64.b64encode(buf3.getvalue()).decode('utf-8')
                plt.close()

                # Store results in session
                request.session['results'] = {
                    'theta': theta.tolist(),
                    'performance': performance,
                    'learning_curve_image': learning_curve_image,
                    'model_type': model_type,
                    'learning_rate': learning_rate,
                    'num_iterations': num_iterations,
                    'abs_error_hist': abs_error_hist,
                    'pred_vs_true_scatter': pred_vs_true_scatter,
                    'train_test_curve': train_test_curve
                }
                
                return redirect('core:results')
                
            except Exception as e:
                error_msg = str(e)
                if "No valid numeric data found" in error_msg:
                    messages.error(request,
                        'Erreur d\'algorithme : Aucune donnée numérique valide trouvée. '
                        'Veuillez revenir à l\'étape de préparation des données.')
                elif "can't multiply sequence by non-int" in error_msg or "unsupported operand type" in error_msg:
                    messages.error(request,
                        'Erreur d\'algorithme : Problème de type de données. '
                        'Les données contiennent encore des valeurs non-numériques. '
                        'Veuillez revenir à l\'étape de préparation des données.')
                else:
                    messages.error(request, f'Erreur d\'exécution de l\'algorithme : {error_msg}')
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