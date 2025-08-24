import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import shap
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class Pipeline:
    """
    Lightweight ML pipeline for classification
    Decision Trees with interpretability
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.shap_explainer = None
        
    def load_data(self, X, y, scale_features=True):
        """
        Load prepared feature matrix and target vector
        
        Parameters:
        - X: DataFrame or array with features
        - y: Series or array with target labels
        - scale_features: Whether to scale features (default True)
        """
        
        print(f"Loading data...")
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Store feature names
        self.feature_names = list(X.columns)
        self.scale_features = scale_features
        
        print(f"Dataset: {X.shape[0]} samples × {X.shape[1]} features")
        print(f"Target classes: {sorted(y.unique())}")
        print(f"Class distribution:\n{y.value_counts()}")
        
        return X, y
    
    def create_splits(self, X, y, test_size=0.2, val_size=0.15):
        """
        Create train/validation/test splits
        """
        
        print(f"\n Creating splits...")
        
        # Split into train/temp and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Split temp into train and validation
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=self.random_state, stratify=y_temp
        )
        
        # Scale features if requested
        if self.scale_features:
            print(f"Scaling features...")
            X_train = pd.DataFrame(
                self.scaler.fit_transform(X_train), 
                columns=X_train.columns, 
                index=X_train.index
            )
            X_val = pd.DataFrame(
                self.scaler.transform(X_val), 
                columns=X_val.columns, 
                index=X_val.index
            )
            X_test = pd.DataFrame(
                self.scaler.transform(X_test), 
                columns=X_test.columns, 
                index=X_test.index
            )
        
        print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self, X_train, y_train, X_val, y_val, 
                   max_depth=15, min_samples_split=20, min_samples_leaf=10):
        """
        Train Decision Tree classifier
        """
        
        print(f"\n Training Decision Tree...")
        
        # Initialize model
        self.model = DecisionTreeClassifier(
            random_state=self.random_state,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features='sqrt'
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Validation accuracy
        y_val_pred = self.model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        
        print(f"Validation Accuracy: {val_accuracy:.3f}")
        print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return {
            'val_accuracy': val_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def evaluate_model(self, X_test, y_test, save_results=True, output_dir="./ml_results"):
        """Evaluate model and create SHAP explanations"""
        
        if self.model is None:
            raise ValueError("No model trained. Run train_model() first.")
        
        print(f"\n Evaluating model...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test Accuracy: {test_accuracy:.3f}")
        print(f"\n Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # SHAP analysis
        print(f"\n Generating SHAP explanations...")
        
        try:
            self.shap_explainer = shap.TreeExplainer(self.model)
            shap_values = self.shap_explainer.shap_values(X_test)
            
            # For multiclass, calculate average importance
            if isinstance(shap_values, list):
                shap_importance = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
            else:
                shap_importance = np.abs(shap_values).mean(0)
            
            # Feature importance dataframe
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'shap_importance': shap_importance
            }).sort_values('shap_importance', ascending=False)
            
            print(f"Top 10 Important Features:")
            for i, row in feature_importance.head(10).iterrows():
                print(f"    {row['feature']}: {row['shap_importance']:.4f}")
        
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
            shap_values = None
            feature_importance = None
        
        # Save results
        if save_results:
            self._save_results(y_test, y_pred, feature_importance, output_dir)
        
        return {
            'test_accuracy': test_accuracy,
            'shap_values': shap_values,
            'feature_importance': feature_importance,
            'predictions': y_pred
        }
    
    def _save_results(self, y_test, y_pred, feature_importance, output_dir):
        """
        Save evaluation results to files
        """
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        class_report_df = pd.DataFrame(class_report).transpose()
        class_report_df.to_csv(
            os.path.join(output_dir, f"classification_report_{timestamp}.csv")
        )
        
        # Feature importance
        if feature_importance is not None:
            feature_importance.to_csv(
                os.path.join(output_dir, f"feature_importance_{timestamp}.csv"), 
                index=False
            )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix_df = pd.DataFrame(conf_matrix)
        conf_matrix_df.to_csv(
            os.path.join(output_dir, f"confusion_matrix_{timestamp}.csv")
        )
        
        print(f"Results saved to: {output_dir}")
    
    def plot_shap_summary(self, X_test, max_display=20, save_plot=True, output_dir="./ml_results"):
        """
        Create SHAP summary plots
        """
        
        if self.shap_explainer is None:
            print("No SHAP explainer. Run evaluate_model() first.")
            return
        
        print(f"\nCreating SHAP plots...")
        
        try:
            shap_values = self.shap_explainer.shap_values(X_test)
            
            # Create plots
            fig, axes = plt.subplots(2, 1, figsize=(12, 14))
            
            # Summary plot
            plt.sca(axes[0])
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[0], X_test, max_display=max_display, show=False)
            else:
                shap.summary_plot(shap_values, X_test, max_display=max_display, show=False)
            axes[0].set_title('SHAP Feature Importance Summary')
            
            # Bar plot
            plt.sca(axes[1])
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[0], X_test, plot_type="bar", max_display=max_display, show=False)
            else:
                shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=max_display, show=False)
            axes[1].set_title('SHAP Feature Importance Bar Plot')
            
            plt.tight_layout()
            
            if save_plot:
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_path = os.path.join(output_dir, f"shap_summary_{timestamp}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"SHAP plots saved: {plot_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    def get_feature_importance(self):
        """
        Get scikit-learn feature importance
        """
        
        if self.model is None:
            raise ValueError("No model trained.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

def run_full_pipeline(X, y, scale_features=True, save_results=True, output_dir="./ml_results"):
    """
    Run complete ML pipeline on prepared data
    
    Parameters:
    - X: Feature matrix (DataFrame or array)
    - y: Target labels (Series or array)
    - scale_features: Whether to scale features
    - save_results: Whether to save results to files
    - output_dir: Directory to save results
    
    Returns: Dictionary with results
    """
    
    # Initialize pipeline
    pipeline = Pipeline()
    
    # Load data
    X, y = pipeline.load_data(X, y, scale_features=scale_features)
    
    # Create splits
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.create_splits(X, y)
    
    # Train model
    train_results = pipeline.train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    eval_results = pipeline.evaluate_model(X_test, y_test, save_results=save_results, output_dir=output_dir)
    
    # Create SHAP plots
    pipeline.plot_shap_summary(X_test, save_plot=save_results, output_dir=output_dir)
    
    # Get sklearn feature importance
    sklearn_importance = pipeline.get_feature_importance()
    
    return {
        'pipeline': pipeline,
        'test_accuracy': eval_results['test_accuracy'],
        'val_accuracy': train_results['val_accuracy'],
        'cv_accuracy': train_results['cv_mean'],
        'feature_importance_shap': eval_results['feature_importance'],
        'feature_importance_sklearn': sklearn_importance,
        'predictions': eval_results['predictions'],
        'splits': {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test
        }
    }
